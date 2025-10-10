# 🚀 Transformer Large Language Model

A simple yet powerful Transformer LLM implementation built with PyTorch.

---

## ✨ Features

### What This Code Does
- 🔤 **Train BPE tokenizer from scratch** - Build your own vocabulary
- 📥 **Load Mistral tokenizer** - Use pre-trained Mistral BPE tokenizer
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

---

## 🔤 Tokenizer Setup

You have two options for tokenization:

### Option A: Train Your Own Tokenizer (Recommended for Custom Vocabularies)
```bash
python train_tokenizer.py
```

**Tokenizer Training Options:**

| Parameter                      | Description                            |
| ------------------------------ | -------------------------------------- |
| `--dataset_path_huggingface`   | HuggingFace dataset path               |
| `--dataset_sub_set`            | Subset of the dataset to use           |
| `--tokenizer_train_shard_size` | Number of examples per training shard  |
| `--trust_remote_code`          | Trust remote code when loading dataset |

**Pre-trained Resources (Custom Tokenizer Only):**
- 📦 [Custom BPE Tokenizer on Kaggle](https://www.kaggle.com/models/rohankhanbd/bpetokenizer) - Download to skip training
- 📊 [Pre-tokenized FineWeb-Edu Dataset](https://www.kaggle.com/datasets/rohankhanbd/half-tokenized-fineweb-edu-10b-subset) - Already tokenized with the custom tokenizer above

> **⚠️ Important:** These pre-trained resources only work with the custom tokenizer workflow (Option A). They are **not compatible** with Mistral's tokenizer.

### Option B: Use Mistral's Pre-trained Tokenizer (Fast Start)
Skip the tokenizer training step and use the `--load_mistral_tokenizer=True` flag in subsequent steps. This leverages Mistral's proven vocabulary and is great for quick experimentation.

**Setup:** Download only the `tokenizer.json` file (not the model weights) from [Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/tree/main) and place it in your project directory.

> **💡 Tip:** Using Mistral's tokenizer means you can skip step 1 entirely and start directly with data tokenization. However, you'll need to tokenize your own dataset from scratch.

---

## 📊 Training Pipeline

### 1️⃣ Tokenize Your Dataset  
```bash
# With custom tokenizer
python tokenize_data.py

# OR with Mistral tokenizer
python tokenize_data.py --load_mistral_tokenizer=True
```

**Data Tokenization Options:**

| Parameter                      | Description                                                |
| ------------------------------ | ---------------------------------------------------------- |
| `--dataset_path_huggingface`   | HuggingFace dataset path                                   |
| `--dataset_sub_set`            | Subset of the dataset to use                               |
| `--data_file_name`             | Output filename for tokenized data                         |
| `--encoded_dataset_shard_size` | Number of examples per shard                               |
| `--load_mistral_tokenizer`     | Use Mistral's tokenizer instead of custom (default: False) |

**Pre-tokenized Dataset (Custom Tokenizer Only):**
- 📊 [FineWeb-Edu 10B Subset](https://www.kaggle.com/datasets/rohankhanbd/half-tokenized-fineweb-edu-10b-subset) - Skip tokenization if using the custom tokenizer

> **⚠️ Note:** The pre-tokenized dataset above only works with the custom tokenizer, not with `--load_mistral_tokenizer`.

### 2️⃣ Train the Model
```bash
# With custom tokenizer
python train.py

# OR with Mistral tokenizer
python train.py --load_mistral_tokenizer=True
```

**Training Options:**

| Parameter                  | Description                                                |
| -------------------------- | ---------------------------------------------------------- |
| `--steps`                  | Total training steps                                       |
| `--eval_rate`              | Evaluate every N steps                                     |
| `--eval_steps`             | Number of evaluation steps                                 |
| `--save_rate`              | Save checkpoint every N steps                              |
| `--warm_up`                | Learning rate warmup steps                                 |
| `--total_batch_size`       | Total batch size across all GPUs                           |
| `--batch_size`             | Batch size per GPU                                         |
| `--seed`                   | Random seed for reproducibility                            |
| `--lr`                     | Peak learning rate                                         |
| `--min_lr`                 | Minimum learning rate                                      |
| `--weight_decay`           | Weight decay for regularization                            |
| `--beta1`                  | Adam beta1 parameter                                       |
| `--beta2`                  | Adam beta2 parameter                                       |
| `--backend`                | Distributed backend (nccl/gloo)                            |
| `--save_file_name`         | Checkpoint filename                                        |
| `--data_file_name`         | Tokenized data filename                                    |
| `--compile_model`          | Use PyTorch 2.0 compilation                                |
| `--load_mistral_tokenizer` | Use Mistral's tokenizer instead of custom (default: False) |

### 3️⃣ Generate Text
```bash
# With custom tokenizer
python generate.py --input_text "Hello" --num_tokens_to_generate 20

# OR with Mistral tokenizer
python generate.py --input_text "Hello" --num_tokens_to_generate 20 --load_mistral_tokenizer=True
```

**Generation Options:**

| Parameter                  | Description                                                |
| -------------------------- | ---------------------------------------------------------- |
| `--input_text`             | Starting text for generation                               |
| `--num_tokens_to_generate` | Number of tokens to generate                               |
| `--temperature`            | Randomness control (0.0-2.0, higher = more random)         |
| `--top_p`                  | Nucleus sampling threshold (0.0-1.0)                       |
| `--save_file_name`         | Model checkpoint filename to load                          |
| `--load_mistral_tokenizer` | Use Mistral's tokenizer instead of custom (default: False) |

**Examples:**
```bash
# With custom tokenizer
python generate.py --input_text "Once upon a time" --num_tokens_to_generate 50 --temperature 0.7 --top_p 0.9

# With Mistral tokenizer
python generate.py --input_text "Once upon a time" --num_tokens_to_generate 50 --temperature 0.7 --top_p 0.9 --load_mistral_tokenizer=True
```

---

## ⚠️ Important Notes

### Tokenizer Consistency
**Critical:** You must use the same tokenizer for training and generation that was used for data tokenization. 

- ✅ If you tokenized data with `--load_mistral_tokenizer`, use it for training and generation
- ✅ If you tokenized data with your custom tokenizer, don't use `--load_mistral_tokenizer` flag
- ❌ Mixing tokenizers will cause errors or produce gibberish output

### Which Tokenizer Should I Use?

| Use Case                | Recommendation                                 |
| ----------------------- | ---------------------------------------------- |
| 🚀 Quick experimentation | Mistral tokenizer (`--load_mistral_tokenizer`) |
| 🔬 Research & learning   | Custom tokenizer (train your own)              |
| 🌍 Non-English languages | Custom tokenizer trained on your data          |
| 📚 Domain-specific text  | Custom tokenizer trained on domain data        |
| ⚡ Fast prototyping      | Mistral tokenizer (`--load_mistral_tokenizer`) |

---

## 🏗️ Architecture Highlights

- **🔥 DeepSeek Multi-Head Latent Attention** - Enhanced attention mechanism
- **⚖️ Mixture-of-Experts** - Scalable expert routing
- **⚡ PyTorch DDP** - Efficient multi-GPU orchestration
- **🎯 Mixed Precision** - FP16/BF16 training optimization

---

## 📝 Example Workflows

### Workflow 1: Using Mistral Tokenizer (Fastest)
```bash
# Step 1: Tokenize data
python tokenize_data.py --load_mistral_tokenizer=True

# Step 2: Train model
python train.py --load_mistral_tokenizer=True --steps 10000

# Step 3: Generate text
python generate.py --input_text "Hello world" --load_mistral_tokenizer=True
```

### Workflow 2: Using Custom Tokenizer (Most Flexible)
```bash
# Step 1: Train tokenizer
python train_tokenizer.py

# Step 2: Tokenize data
python tokenize_data.py

# Step 3: Train model
python train.py --steps 10000

# Step 4: Generate text
python generate.py --input_text "Hello world"
```

---