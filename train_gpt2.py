from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import inspect
import math
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import tiktoken
import time
from hellaswag import render_example, get_most_likely_row,iterate_examples

@dataclass
class GPTConfig:
    vocab_size: int = 50257 #number of tokens in the vocabulary with 50000 merges, 256 bytes for each token, and 1 for special token <|endoftext|>
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.scale = self.head_size ** -0.5
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        # q, k, v = [t.view(B, T, self.n_head, self.head_size).transpose(1, 2) for t in qkv]
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        # 
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True) #this is flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # creates a memory to the gpu (contiguos) without this y is never saved to the memory
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        """ 
        In a Transformer model, the Residual Stream acts like a long highway where data flows from the input to the output. 
        As data travels down this highway, each new Transformer layer calculates changes (via its Attention and MLP blocks) and adds them to the stream:(\text{Output}=\text{Input}+\text{Layer}_{1}+\text{Layer}_{2}+\dots +\text{Layer}_{N}\)
        Mathematically, when you add independent random variables together, their variances add up.
        If every layer has an output variance of \(1\), after passing through \(N\) layers, the variance of your data stream blows up to \(N\).
        Because a standard Transformer block has two sub-layers that add to the residual stream (1. Attention and 2. MLP), a model with \(N\) layers adds exactly \(2N\) signals to the stream.
        By the time the data reaches the final layer of a deep model, the numbers have grown massive. 
        This destabilizes the gradients, causes training to crash, and makes the model stop learning.
        2. 
        The Solution: The \(\frac{1}{\sqrt{2L}}\) ScaleTo keep the variance of the residual stream perfectly balanced at exactly \(1.0\) from start to finish, we must shrink the initialization size of the weights in proportion to how deep the network is.
        According to probability theory, if you want the sum of \(2N\) elements to have a variance of \(1\), each individual element must be scaled down by the square root of the total additions:\(\text{Scaling\ Factor}=\frac{1}{\sqrt{2\times \text{number\ of\ layers}}}\)
        In your PyTorch code, (2 * self.config.n_layer) ** -0.5 is the exact math expression for \(\frac{1}{\sqrt{2L}}\).

        """

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std= 0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_type} from HuggingFace Transformers library...")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['c_attn.weight', 'c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"Number of keys in our model ({len(sd_keys)}) does not match number of keys in HuggingFace model ({len(sd_keys_hf)})"  
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for key {k}: HuggingFace shape {sd_hf[k].shape} vs our model shape {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for key {k}: HuggingFace shape {sd_hf[k].shape} vs our model shape {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    

with open('C:\\Users\\ABDALAH\\Downloads\\input.txt', 'r') as f:
    text = f.read()

import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt
class DataLoaderLite: ##**
    def __init__(self, B, T, process_rank, num_processes, split):

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        # with open('C:\\Users\\ABDALAH\\Downloads\\input.txt', 'r') as f:
        #     text = f.read()
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards)>0, f'no shards found for split{split}'
        if master_process:
            print(f'found{len(shards)} shards for split {split}')

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

        # self.enc = tiktoken.get_encoding('gpt2')
        # self.tokens = self.enc.encode(text)
        # self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        # print(f"Total tokens in dataset: {len(self.tokens)}")
        # self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T*self.num_processes
        if self.current_position + B*T*self.num_processes + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = self.B * self.T * self.process_rank
        return x, y
    
learning_rate = 6e-4
min_lr = learning_rate * 0.1
warmup_iters = 715
max_steps = 19073 #multiply by 4 due to training takes around 1.5hrs



enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode(text)
# B, T = 4, 32
# buff = torch.tensor(tokens[:B*T + 1], dtype=torch.long)
# x = buff[:-1].view(B, T)
# y = buff[1:].view(B, T)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_steps - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0
    master_process = True
    ddp_local_rank = 0
    ddp_world_size = 1

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')

torch.manual_seed(13337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(13337)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, 'make sure total_batch_size is divisible by B*T'
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f'Total desired batch size:{total_batch_size}')
    print(f'calculated gradient accumulation steps: {grad_accum_steps}')

print('I am GPU', ddp_rank)
import sys; sys.exit(0)
train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train') 
val_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val') 
torch.set_float32_matmul_precision('high')

num_return_sequences = 5
max_length = 30


model = GPT(GPTConfig(vocab_size= 50304))
use_compile = False
if use_compile:
    model = torch.compile(model)
if DDP:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

#model = torch.compile(model) |this makes runs faster but requires Microsoft Visual C++ (MSVC) compiler toolset
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=10**-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device_type='cpu')

log_dir = 'log'
os.makedirs(log_dir, exist_ok= True)
log_file = os.path.join(log_dir, f'log.txt')
with open(log_file, 'w') as f:
    pass

for step in range(max_steps):
    loss_accum = 0.0
    t0 = time.time()
    last_step = (step == max_steps - 1)
    if step % 100 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device = 'cpu', dtype= torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)
        if master_process:
            print(f'validation loss{val_loss_accum.item():.4f}')
            with open(log_file, 'a') as f:
                f.write(f'{step} val {val_loss_accum.item():.4f}\n')
            if step>0 and (step%5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f'model_{step:05d}.pt')
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)


    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                with torch.autocast (device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.Reduce0p.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: fnum_correct_norm)/fnum_total)=facc_norm:.4f)")
                with open(log_file, "a") as f:
                    f.write(f'{step} hella {acc_norm:.4f}\n')

    if step > 0 and step % 100 == 0 and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode('Hello, I \'am a language model,')
        tokens = torch.tensor(tokens, dtype= torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device = device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)
                logits = logits[:,-1,:]
                probs = F.softmax(logits, dim = -1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
                next_token = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, next_token)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(">", decoded)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        optimizer.zero_grad()
        with torch.autocast(device_type= 'cpu', dtype = torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1) #prevents shocks when loss return very large value
    optimizer.step()
    #torch.cuda.synchronize() when there is cuda and it is used to wait until gpu operations are fully done
    t1 = time.time()
    dt = (t1-t0)*1000 #time in milliseconds
    tokens_per_second = (train_loader.B*train_loader.T * grad_accum_steps*ddp_world_size)/(t1-t0)
    if master_process:
        print(f"Iteration {step}, loss: {loss_accum.item()} dt: {dt:.2f}ms, norm: {norm} tok/sec {tokens_per_second}")
        with open(log_file, 'a') as f:
            f.write(f'{step} train {loss_accum.item():.6f}\n')
if ddp:
    destroy_process_group()


# torch.manual_seed(42)
# while x.size(1) < max_length:
#     logits, loss = model(x, y)
#     logits = logits[:, -1, :]
#     probs = F.softmax(logits, dim=-1)
#     topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
#     next_token = torch.multinomial(topk_probs, num_samples=1)
#     xcol = torch.gather(topk_indices, -1, next_token)
#     x = torch.cat((x, xcol), dim=1)

# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)

#=====educational purposes only===========
"""
what is Gradient Clipping?
Gradient clipping forces gradients to stay within a safe range.
Instead of letting them explode:
we scale them down if they are too large

====torch.topk========
what is it?
forexample below of how topk works, it returns the top k values and their corresponding indices from a tensor. 
In this case, it returns the top 3 values from the tensor x and their corresponding indices. 
The output shows that the top 3 values are 5, 4, and 3, and their corresponding indices are 4, 3, and 2 respectively.
x = torch.tensor([1., 2., 3., 4., 5.])
values, indices = torch.topk(x, 3)
print(values)  # tensor([5., 4., 3.])
print(indices) # tensor([4, 3, 2])

=====torch.gather=====
what is it?
torch.gather is a function that gathers values along an axis specified by dim.
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

idx = torch.tensor([
    [0, 2],
    [1, 0]
])
result = torch.gather(x, dim=1, index=idx)
print(result)
tensor([
    [10, 30],
    [50, 40]
])
what is sys?
is a built-in Python tool that helps your program talk to the computer's operating system.
Think of it like a remote control for your computer.
It allows you to manipulate the Python runtime environment, such as handling command-line arguments, exiting the program, and accessing standard input/output streams.
forexample,
import sys
print(sys.argv)  # prints the list of command-line arguments passed to the script
forexample;
import sys
# If you run: python script.py hello world
print(sys.argv)
# Output: ['script.py', 'hello', 'world']
# It captures what you typed after the filename
sys.exit()  # exits the program
sys.executable  # returns the path of the Python interpreter executable
sys.version  # returns the version of Python being used
sys.path  # returns a list of directories that the Python interpreter searches for modules
When Python needs a module/library, it asks: "Where do I find modules?"
Python's answer is sys.path - a list of folders.

.all()
this check elements equalness inside tensors

.data_ptr()
this checks data pointer

hasattr()
checks if an object has a specific attribute or method.
forexample
class Dog:
    def __init__(self):
        self.name = "Buddy"
        self.age = 5
    
    def bark(self):
        print("Woof!")

dog = Dog()

# Check if dog has 'name' attribute
print(hasattr(dog, 'name'))  # ✅ True (it exists)

# Check if dog has 'color' attribute
print(hasattr(dog, 'color'))  # ❌ False (doesn't exist)

# Check if dog has 'bark' method
print(hasattr(dog, 'bark'))  # ✅ True (method exists)

# Check if dog has 'fly' method
print(hasattr(dog, 'fly'))  # ❌ False (doesn't exist)

import code; code.interact(local = local())

source----Chatgpt-----floating points and other below------

floating points
torch.backends.cuda.matmul.allow_tf32 = True --->this activates gpu to use TF32
torch.set_float32_matmul_precision('high')
this is done internally on the gpu and not outside it

what is autocast?
automatic mixed precision for forward pass
It automatically chooses lower precision (FP16/BF16) where safe
Convert some operations to FP16/BF16:
->matrix multiplications
->convolutions
Keep some in FP32:
->softmax (sometimes)
->layernorm
->reductions (for stability)
===============================
model weights stay FP32
but computations may run in FP16/BF16

Gradscaler
when training using fp16 gradients may become very small so this prevents from becoming that 
since leaving behind that will make gradients to approximate to zero which is not essential during training


Warmup is a training strategy where the learning rate starts small and gradually increases to prevent instability at the beginning of training.

Cosine learning rate decay gradually reduces step size so the model can move from fast learning early to stable fine-tuning at convergence.


now when we run our model if there is ddp(distributed data parallel)
we use
--> torchrun --standalone --nproc_per_node = 8(number of gpus) train_gpt2.py(script itself)
what is destroy_process_group?

======DATASET=======
1.slimpajama or redpajama dataset
2.fineweb dataset
3. huggingface have released fineweb-Edu

what is commoncrawl??   
what are shards??
subset of the entire dataset

world_size = int(os.environ["WORLD_SIZE"])
Total number of processes (GPUs) in the entire training job(WORLD_SIZE = size of the team)

rank = int(os.environ["RANK"])
Global ID of this process in the entire distributed system.

local_rank = int(os.environ["LOCAL_RANK"])
GPU index on the current machine only

Example;
| Global Rank | Machine | Local Rank | GPU  |
| ----------- | ------- | ---------- | ---- |
| 0           | Node A  | 0          | GPU0 |
| 1           | Node A  | 1          | GPU1 |
| 2           | Node A  | 2          | GPU2 |
| 3           | Node A  | 3          | GPU3 |
| 4           | Node B  | 0          | GPU0 |
| 5           | Node B  | 1          | GPU1 |
| 6           | Node B  | 2          | GPU2 |
| 7           | Node B  | 3          | GPU3 |


dist.init_process_group(backend="nccl")--> it is used for gpu communication
chatgpt --> section(Fineweb.py Source code)

"""
