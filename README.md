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

| Parameter                      | Description                       |
| ------------------------------ | --------------------------------- |
| `--dataset_path_huggingface`   | HuggingFace dataset path          |
| `--dataset_sub_set`            | Dataset subset to use             |
| `--tokenizer_train_shard_size` | Shard size for tokenizer training |
| `--trust_remote_code`          | Trust remote code execution       |

#### 2️⃣ Tokenize Your Dataset  
```bash
python tokenize_data.py
```
*Or use our pre-tokenized dataset: [FineWeb-Edu 10B Subset](https://www.kaggle.com/datasets/rohankhanbd/half-tokenized-fineweb-edu-10b-subset)*

**Data Tokenization Options:**

| Parameter                      | Description                    |
| ------------------------------ | ------------------------------ |
| `--dataset_path_huggingface`   | HuggingFace dataset path       |
| `--dataset_sub_set`            | Dataset subset to use          |
| `--data_file_name`             | Name of the data file          |
| `--encoded_dataset_shard_size` | Shard size for encoded dataset |

#### 3️⃣ Train the Model
```bash
python train.py
```

**Training Options:**

| Parameter            | Description                      |
| -------------------- | -------------------------------- |
| `--steps`            | Number of training steps         |
| `--eval_rate`        | Evaluation frequency             |
| `--eval_steps`       | Number of evaluation steps       |
| `--save_rate`        | Checkpoint save frequency        |
| `--warm_up`          | Learning rate warmup steps       |
| `--total_batch_size` | Total batch size across all GPUs |
| `--batch_size`       | Per-device batch size            |
| `--seed`             | Random seed for reproducibility  |
| `--lr`               | Learning rate                    |
| `--min_lr`           | Minimum learning rate            |
| `--weight_decay`     | Weight decay coefficient         |
| `--beta1`            | Adam beta1 parameter             |
| `--beta2`            | Adam beta2 parameter             |
| `--backend`          | Distributed backend (e.g., nccl) |
| `--save_file_name`   | Checkpoint save filename         |
| `--data_file_name`   | Training data filename           |
| `--compile_model`    | Enable model compilation         |

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