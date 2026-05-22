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

**Setup:** Download only the `tokenizer.json` from **[Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/tree/main)** and place it in your project directory.

> 💡 **Skip tokenizer training entirely.** Ideal for production or rapid prototyping.

---

### **Option B: Train Your Own Tokenizer**

```bash
python -m scripts.train_tokenizer
```

**Pre-trained Resources (Custom Tokenizer Only):**

* 📦 [Custom BPE Tokenizer (Kaggle)](https://www.kaggle.com/models/rohankhanbd/lilbpetokenizer)
* 📊 [Pre-tokenized FineWeb-Edu Dataset](https://www.kaggle.com/datasets/rohankhanbd/lil-fineweb-dataset)

> ⚠️ These resources only work with the custom tokenizer, **not** Mistral.

---

## 📊 Training Pipeline

### **Pre-training Workflow**

---

#### **1️⃣ Tokenize Your Dataset**

```bash
python -m scripts.tokenize_data
# or
python -m scripts.tokenize_data --load_mistral_tokenizer
```

**Pre-tokenized Dataset (Custom Only):** [FineWeb-Edu 10B subset](https://www.kaggle.com/datasets/rohankhanbd/lil-fineweb-dataset).

> ⚠️ Only compatible with the custom tokenizer.

---

#### **2️⃣ Train the Model**

```bash
python -m scripts.train --compile_model --use_autocast
# or
python -m scripts.train --compile_model --use_autocast --load_mistral_tokenizer
```

---

#### **3️⃣ Generate Text**

```bash
python -m scripts.generate --input_text "Hello" --num_tokens_to_generate 20 --compile_model
# or
python -m scripts.generate --input_text "Hello" --num_tokens_to_generate 20 --load_mistral_tokenizer --compile_model
```

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
python -m scripts.tokenize_sft_data
# or
python -m scripts.tokenize_sft_data --load_mistral_tokenizer
```

---

#### **2️⃣ Fine-tune the Model**

```bash
python -m scripts.train_sft --compile_model --use_autocast
# or
python -m scripts.train_sft --load_mistral_tokenizer --compile_model --use_autocast
```

---

#### **3️⃣ Test Your Instruction Model**

```bash
python -m scripts.generate --input_text "Python is" --num_tokens_to_generate 100 --save_file_name lilgpt_inst --compile_model
# or
python -m scripts.generate --input_text "Python is" --num_tokens_to_generate 100 --save_file_name lilgpt_inst --load_mistral_tokenizer --compile_model
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
