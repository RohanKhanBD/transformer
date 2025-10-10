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

### Option A: Use Mistral's Pre-trained Tokenizer (Recommended)
Skip the tokenizer training step and use the `--load_mistral_tokenizer=True` flag in subsequent steps. This leverages Mistral's proven vocabulary and is great for quick experimentation and production use.

**Setup:** Download only the `tokenizer.json` file (not the model weights) from [Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/tree/main) and place it in your project directory.

> **💡 Tip:** Using Mistral's tokenizer means you can skip tokenizer training entirely and start directly with data tokenization!

### Option B: Train Your Own Tokenizer (For Custom Vocabularies)
```bash
python train_tokenizer.py
```

**Tokenizer Training Options:**

| Parameter                      |
| ------------------------------ |
| `--dataset_path_huggingface`   |
| `--dataset_sub_set`            |
| `--tokenizer_train_shard_size` |
| `--trust_remote_code`          |

**Pre-trained Resources (Custom Tokenizer Only):**
- 📦 [Custom BPE Tokenizer on Kaggle](https://www.kaggle.com/models/rohankhanbd/bpetokenizer) - Download to skip training
- 📊 [Pre-tokenized FineWeb-Edu Dataset](https://www.kaggle.com/datasets/rohankhanbd/half-tokenized-fineweb-edu-10b-subset) - Already tokenized with the custom tokenizer above

> **⚠️ Important:** These pre-trained resources only work with the custom tokenizer workflow (Option B). They are **not compatible** with Mistral's tokenizer.

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

| Parameter                      |
| ------------------------------ |
| `--dataset_path_huggingface`   |
| `--dataset_sub_set`            |
| `--data_file_name`             |
| `--encoded_dataset_shard_size` |
| `--load_mistral_tokenizer`     |

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

| Parameter                  |
| -------------------------- |
| `--steps`                  |
| `--eval_rate`              |
| `--eval_steps`             |
| `--save_rate`              |
| `--warm_up`                |
| `--total_batch_size`       |
| `--batch_size`             |
| `--seed`                   |
| `--lr`                     |
| `--min_lr`                 |
| `--weight_decay`           |
| `--beta1`                  |
| `--beta2`                  |
| `--backend`                |
| `--save_file_name`         |
| `--data_file_name`         |
| `--compile_model`          |
| `--load_mistral_tokenizer` |

### 3️⃣ Generate Text
```bash
# With custom tokenizer
python generate.py --input_text "Hello" --num_tokens_to_generate 20

# OR with Mistral tokenizer
python generate.py --input_text "Hello" --num_tokens_to_generate 20 --load_mistral_tokenizer=True
```

**Generation Options:**

| Parameter                  |
| -------------------------- |
| `--input_text`             |
| `--num_tokens_to_generate` |
| `--temperature`            |
| `--top_p`                  |
| `--save_file_name`         |
| `--load_mistral_tokenizer` |

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

| Use Case                | Recommendation                          |
| ----------------------- | --------------------------------------- |
| 🚀 Most users            | **Mistral tokenizer** (recommended)     |
| ⚡ Quick experimentation | **Mistral tokenizer**                   |
| 🏭 Production use        | **Mistral tokenizer**                   |
| 🔬 Research & learning   | Custom tokenizer                        |
| 🌍 Non-English languages | Custom tokenizer trained on your data   |
| 📚 Domain-specific text  | Custom tokenizer trained on domain data |

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