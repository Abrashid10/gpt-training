# hellaswag.py - Download and prepare dataset

import os
import json
import requests
import tiktoken
import torch

DATA_CACHE_DIR = "cache"
enc = tiktoken.get_encoding("gpt2")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def download_file(url, filename):
    """Download file from URL"""
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        f.write(response.content)

def download(split):
    """Download HellaSwag dataset"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    Render example into tokens, mask, and label
    - tokens: 4xN (4 possible endings)
    - mask: marks where completion is (for loss calculation)
    - label: correct answer index (0-3)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]
    
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }
    
    # Tokenize context
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    
    tok_rows = []
    mask_rows = []
    
    # Tokenize each of the 4 endings
    for end in endings:
        end_tokens = enc.encode(" " + end)  # prepend space for GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)
    
    # Pad all rows to same length
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    
    return data, tokens, mask, label

def iterate_examples(split):
    """Load and iterate through HellaSwag examples"""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

def get_most_likely_row(tokens, mask, logits):
    """
    Calculate loss for each of 4 endings, return index with lowest loss
    (lowest loss = most likely completion)
    """
    # Shift logits and tokens for autoregressive prediction
    shift_logits = (logits[..., :-1, :]).contiguous()  # predictions
    shift_tokens = (tokens[..., 1:]).contiguous()       # targets (shifted)
    
    # Flatten for cross entropy
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    
    # Calculate loss at each position
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)  # reshape back to (4, seq_len)
    
    # Only calculate loss on the completion (where mask == 1)
    shift_mask = (mask[..., 1:]).contiguous()  # shift mask to match shifted tokens
    masked_shift_losses = shift_losses * shift_mask
    
    # Average loss per completion
    sum_loss = masked_shift_losses.sum(dim=1)           # sum loss per row
    avg_loss = sum_loss / shift_mask.sum(dim=1)         # divide by number of tokens
    
    # Return index of completion with lowest loss
    pred_norm = avg_loss.argmin().item()
    return pred_norm