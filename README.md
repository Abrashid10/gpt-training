[README.md](https://github.com/user-attachments/files/33706807/README.md)
# GPT Training

Training a GPT-2 style language model from scratch, with implementations in **PyTorch**, **plain C**, and **CUDA**, plus experiments with agents and MCP tooling.

> **Status:** Work in progress

## Overview

This repository explores how GPT-style language models are trained at different levels of abstraction:

- **Python / PyTorch** reference implementation for clarity and fast experimentation
- **Plain C** implementation for understanding every step with no dependencies
- **CUDA** implementations (mixed precision and FP32) for GPU-accelerated training
- **Agents and MCP** experiments built around the training workflow
- **Kaggle** notebooks/scripts for running training on free GPU resources

The small [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset is used so you can see results quickly.

## Repository Structure

```
gpt-training/
├── agents/                  # Agent experiments
├── dev/data/tinyshakespeare/# Tiny Shakespeare dataset and preparation files
├── kaggle/                  # Kaggle training setup
├── llmc/                    # Shared C/CUDA helper code
├── mcp/                     # MCP (Model Context Protocol) experiments
├── train gpt/               # Additional training files
├── data_common.py           # Shared data utilities
├── tinyshakespeare.py       # Downloads and tokenizes the Tiny Shakespeare dataset
├── train_gpt.py             # PyTorch training script
├── train_gpt2.c             # Pure C training implementation
├── train_gpt.cu             # CUDA training (mixed precision)
├── train_gpt_fp32.cu        # CUDA training (FP32)
├── test_gpt.cu              # CUDA test for the mixed precision version
└── test_gpt_fp32.cu         # CUDA test for the FP32 version
```

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python    | 3.8+ with `torch`, `numpy`, `tqdm`, `requests` |
| C         | `gcc` or `clang` (OpenMP recommended) |
| CUDA      | NVIDIA GPU, CUDA toolkit, `nvcc` |

Install the Python dependencies:

```bash
pip install torch numpy tqdm requests
```

## Quick Start

### 1. Prepare the dataset

```bash
python tinyshakespeare.py
```

### 2. Train with PyTorch

```bash
python train_gpt.py
```

### 3. Train with plain C (CPU)

```bash
gcc -O3 -fopenmp -o train_gpt2 train_gpt2.c -lm
OMP_NUM_THREADS=8 ./train_gpt2
```

### 4. Train with CUDA (GPU)

```bash
# Mixed precision
nvcc -O3 --use_fast_math -o train_gpt train_gpt.cu
./train_gpt

# FP32
nvcc -O3 --use_fast_math -o train_gpt_fp32 train_gpt_fp32.cu
./train_gpt_fp32
```

### 5. Run the tests

```bash
nvcc -O3 -o test_gpt test_gpt.cu && ./test_gpt
nvcc -O3 -o test_gpt_fp32 test_gpt_fp32.cu && ./test_gpt_fp32
```

> Adjust compile flags and arguments to match your setup and the options defined in each source file.

## Running on Kaggle

The `kaggle/` folder contains the setup for training on Kaggle's free GPUs. Upload the files, enable a GPU accelerator in the notebook settings, and run the training script.

## Roadmap

- [ ] Add training results and loss curves
- [ ] Document command-line arguments for each script
- [ ] Add text generation / inference examples
- [ ] Document the `agents/` and `mcp/` experiments

## Acknowledgements

The C and CUDA training approach follows the style of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c). Tiny Shakespeare comes from his [char-rnn](https://github.com/karpathy/char-rnn) project.

## Author

**Abdallah**  
GitHub: [@Abrashid10](https://github.com/Abrashid10)

## License

Add a license of your choice (for example MIT) and update this section.
