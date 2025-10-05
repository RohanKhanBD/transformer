# 🚀 Transformer Large Language Model

A simple yet powerful Transformer LLM implementation built with PyTorch.

---

## ✨ Features

### What This Code Does
- 🔤 **Train BPE tokenizer from scratch** - Build your own vocabulary
- 📊 **Dataset tokenization** - Process data with pre-trained BPE tokenizer  
- 🧠 **Transformer training** - Train models from the ground up
- 💬 **Text generation** - Generate text from pre-trained models
- 🔀 **Mixture-of-Experts** - Efficient scaling with MoE architecture
- 🎯 **Multi-Head Latent Attention** - Advanced attention mechanism from DeepSeek
- 🖥️ **Multi-GPU training** - Distributed training with PyTorch DDP
- ⚡ **Mixed-precision training** - Faster training with reduced memory usage

### Current Limitations
- ❌ No HuggingFace model loading support
- ❌ No RLHF fine-tuning capabilities  
- ❌ BPE tokenization only (no other algorithms)
- ❌ No safetensors support
- ❌ Many other features not yet implemented

---

## 🛠️ Quick Start

### Prerequisites
First, install the required dependencies:
```bash
pip install -r requirements.txt
```

> **Note:** All scripts have sensible default values set for their parameters. You can run them without any arguments to get started quickly, or customize the behavior using the options shown below.

### Training Pipeline

#### 1️⃣ Train the Tokenizer
```bash
python train_tokenizer.py
```
*Or use our pre-trained tokenizer: [BPE Tokenizer on Kaggle](https://www.kaggle.com/models/rohankhanbd/bpetokenizer)*

**Tokenizer Training Options:**

| Parameter                      |
| ------------------------------ |
| `--dataset_path_huggingface`   |
| `--dataset_sub_set`            |
| `--tokenizer_train_shard_size` |
| `--trust_remote_code`          |

#### 2️⃣ Tokenize Your Dataset  
```bash
python tokenize_data.py
```
*Or use our pre-tokenized dataset: [FineWeb-Edu 10B Subset](https://www.kaggle.com/datasets/rohankhanbd/half-tokenized-fineweb-edu-10b-subset)*

**Data Tokenization Options:**

| Parameter                      |
| ------------------------------ |
| `--dataset_path_huggingface`   |
| `--dataset_sub_set`            |
| `--data_file_name`             |
| `--encoded_dataset_shard_size` |

#### 3️⃣ Train the Model
```bash
python train.py
```

**Training Options:**

| Parameter            |
| -------------------- |
| `--steps`            |
| `--eval_rate`        |
| `--eval_steps`       |
| `--save_rate`        |
| `--warm_up`          |
| `--total_batch_size` |
| `--batch_size`       |
| `--seed`             |
| `--lr`               |
| `--min_lr`           |
| `--weight_decay`     |
| `--beta1`            |
| `--beta2`            |
| `--backend`          |
| `--save_file_name`   |
| `--data_file_name`   |
| `--compile_model`    |

#### 4️⃣ Generate Text
```bash
python generate.py
```

**Generation Options:**

| Parameter                  | Description                  |
| -------------------------- | ---------------------------- |
| `--input_text`             | Starting text for generation |
| `--num_tokens_to_generate` | Number of tokens to generate |
| `--temperature`            | Randomness control (0.0-2.0) |
| `--top_p`                  | Nucleus sampling threshold   |
| `--save_file_name`         | Model checkpoint filename    |

**Example:**
```bash
python generate.py --input_text "Hello" --num_tokens_to_generate 20 --temperature 0.7 --top_p 0.9
```

---

## 🏗️ Architecture Highlights

- **🔥 DeepSeek Multi-Head Latent Attention** - Enhanced attention mechanism
- **⚖️ Mixture-of-Experts** - Scalable expert routing
- **⚡ PyTorch DDP** - Efficient multi-GPU orchestration
- **🎯 Mixed Precision** - FP16/BF16 training optimization

---