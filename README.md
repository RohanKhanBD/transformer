# 🚀 Transformer Large Language Model

A simple yet powerful Transformer LLM implementation built with **PyTorch**, designed for clarity, modularity, and extensibility.

---

## ✨ Features

### **What This Code Does**

* 🔤 **Train BPE tokenizer from scratch** — Build your own vocabulary
* 🔥 **Load Mistral tokenizer** — Use a proven, production-ready BPE tokenizer
* 📊 **Dataset tokenization** — Efficient pre-processing for large-scale data
* 🧠 **Transformer training** — Train models from the ground up
* 💬 **Text generation** — Generate text using trained checkpoints
* 🎓 **Supervised Fine-Tuning (SFT)** — Instruction / chat model fine-tuning
* 🔀 **Mixture-of-Experts** — Efficient scaling via MoE routing
* 🎯 **Multi-Head Latent Attention** — DeepSeek-inspired attention mechanism
* 🖥️ **Multi-GPU training** — Distributed training using PyTorch DDP
* ⚡ **Mixed-precision training** — FP16/BF16 speed-ups with less memory

### **Current Limitations**

* ❌ No HuggingFace model loading
* ❌ No RLHF pipeline
* ❌ BPE-only tokenization
* ❌ No safetensors support
* ❌ Many advanced features still in progress

---

## 🛠️ Quick Start

### **Prerequisites**

Install required dependencies:

```bash
pip install -r requirements.txt
```

> **Tip:** All scripts include sensible defaults—run them without arguments to get started fast.

---

## 🔤 Tokenizer Setup

Choose one of the two paths:

---

### **Option A: Use Mistral's Pre-trained Tokenizer**

Use the flag `--load_mistral_tokenizer` in training and generation steps.

**Setup:** Download only the `tokenizer.json` from **Mistral-Nemo-Base-2407** and place it in your project directory.

> 💡 **Skip tokenizer training entirely.** Ideal for production or rapid prototyping.

---

### **Option B: Train Your Own Tokenizer**

```bash
python train_tokenizer.py
```

**Tokenizer Training Options:**

| Parameter                      |
| ------------------------------ |
| `--dataset_path_huggingface`   |
| `--dataset_sub_set`            |
| `--tokenizer_file_name`        |
| `--tokenizer_train_shard_size` |
| `--trust_remote_code`          |

**Pre-trained Resources (Custom Tokenizer Only):**

* 📦 Custom BPE Tokenizer (Kaggle)
* 📊 Pre-tokenized FineWeb-Edu Dataset

> ⚠️ These resources only work with the custom tokenizer, **not** Mistral.

---

## 📊 Training Pipeline

### **Pre-training Workflow**

---

#### **1️⃣ Tokenize Your Dataset**

```bash
python tokenize_data.py
# or
python tokenize_data.py --load_mistral_tokenizer
```

**Data Tokenization Options:**

| Parameter                      |
| ------------------------------ |
| `--dataset_path_huggingface`   |
| `--dataset_sub_set`            |
| `--tokenizer_file_name`        |
| `--data_file_name`             |
| `--encoded_dataset_shard_size` |
| `--load_mistral_tokenizer`     |

**Pre-tokenized Dataset (Custom Only):** FineWeb-Edu 10B subset.

> ⚠️ Only compatible with the custom tokenizer.

---

#### **2️⃣ Train the Model**

```bash
python train.py --compile_model --use_autocast
# or
python train.py --compile_model --use_autocast --load_mistral_tokenizer
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
| `--promissed_flops`        |
| `--lr`                     |
| `--min_lr`                 |
| `--weight_decay`           |
| `--beta1`                  |
| `--beta2`                  |
| `--backend`                |
| `--save_file_name`         |
| `--data_file_name`         |
| `--tokenizer_file_name`    |
| `--dtype`                  |
| `--compile_model`          |
| `--use_autocast`           |
| `--load_mistral_tokenizer` |

---

#### **3️⃣ Generate Text**

```bash
python generate.py --input_text "Hello" --num_tokens_to_generate 20 --compile_model
# or
python generate.py --input_text "Hello" --num_tokens_to_generate 20 --load_mistral_tokenizer --compile_model
```

**Generation Options:**

| Parameter                  |
| -------------------------- |
| `--input_text`             |
| `--num_tokens_to_generate` |
| `--temperature`            |
| `--top_p`                  |
| `--save_file_name`         |
| `--backend`                |
| `--tokenizer_file_name`    |
| `--compile_model`          |
| `--load_mistral_tokenizer` |

---

## 🎓 Supervised Fine-Tuning (SFT)

Fine-tune your model to follow instructions or engage in conversation.

### **When to Use SFT**

Perfect for:

* 💬 Chatbots
* 📝 Instruction models
* 🎯 Domain-specific tuning
* 🔄 Behavior alignment

---

### **SFT Workflow**

#### **1️⃣ Prepare the SFT Dataset**

```bash
python tokenize_sft_data.py
# or
python tokenize_sft_data.py --load_mistral_tokenizer
```

**SFT Tokenization Options:**

| Parameter                        |
| -------------------------------- |
| `--sft_dataset_path_huggingface` |
| `--sft_dataset_sub_set`          |
| `--tokenizer_file_name`          |
| `--data_file_name`               |
| `--encoded_dataset_shard_size`   |
| `--load_mistral_tokenizer`       |

---

#### **2️⃣ Fine-tune the Model**

```bash
python train_sft.py --compile_model --use_autocast
# or
python train_sft.py --load_mistral_tokenizer --compile_model --use_autocast
```

**SFT Training Options:**

| Parameter                  |
| -------------------------- |
| `--steps`                  |
| `--eval_rate`              |
| `--eval_steps`             |
| `--save_rate`              |
| `--warm_up`                |
| `--total_batch_size`       |
| `--batch_size`             |
| `--promissed_flops`        |
| `--lr`                     |
| `--min_lr`                 |
| `--weight_decay`           |
| `--beta1`                  |
| `--beta2`                  |
| `--backend`                |
| `--save_file_name`         |
| `--data_file_name`         |
| `--tokenizer_file_name`    |
| `--dtype`                  |
| `--compile_model`          |
| `--use_autocast`           |
| `--load_mistral_tokenizer` |

---

#### **3️⃣ Test Your Instruction Model**

```bash
python generate.py --input_text "Python is" --num_tokens_to_generate 100 --save_file_name lilgpt_inst --compile_model
# or
python generate.py --input_text "Python is" --num_tokens_to_generate 100 --save_file_name lilgpt_inst --load_mistral_tokenizer --compile_model
```

---

## ⚠️ Important Notes

### **Tokenizer Consistency**

You must use the **same tokenizer** for:

* Pre-training
* SFT
* Generation

Mixing tokenizers will break compatibility.

---

### **Which Tokenizer Should You Use?**

| Use Case                  | Recommended Option                        |
| ------------------------- | ----------------------------------------- |
| 🚀 Most users              | Custom tokenizer                          |
| ⚡ Quick testing           | Mistral OR custom                         |
| 🏭 Production              | Mistral tokenizer                         |
| 🎓 SFT / Chat models       | Mistral OR custom (better special tokens) |
| 🔬 Research / learning     | Custom tokenizer                          |
| 🌍 Non-English text        | Custom tokenizer                          |
| 📚 Domain-specific content | Custom tokenizer                          |

---

## 🗺️ Architecture Highlights

* 🔥 DeepSeek Multi-Head Latent Attention
* ⚖️ Mixture-of-Experts
* ⚡ PyTorch Distributed Data Parallel
* 🎯 Mixed Precision (FP16/BF16)

---

## 🤝 Contributing

Contributions are welcome!

* 🐛 Bug reports
* 💡 Feature ideas
* 🔧 Pull requests
* 📖 Documentation improvements

---

## 📄 License

**GNU Affero General Public License (AGPL).**

---

## 🙏 Acknowledgments

* Mistral AI
* HuggingFace
* DeepSeek
* PyTorch Team
