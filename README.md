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

### Training Pipeline

#### 1️⃣ Train the Tokenizer
```bash
python train_tokenizer.py
```
*Or use our pre-trained tokenizer: [BPE Tokenizer on Kaggle](https://www.kaggle.com/models/rohankhanbd/bpetokenizer)*

#### 2️⃣ Tokenize Your Dataset  
```bash
python tokenize_data.py
```
*Or use our pre-tokenized dataset: [FineWeb-Edu 10B Subset](https://www.kaggle.com/datasets/rohankhanbd/half-tokenized-fineweb-edu-10b-subset)*

#### 3️⃣ Train the Model
```bash
python train.py
```

#### 4️⃣ Generate Text
```bash
python generate.py "<input_text>" <num_tokens> <temperature> <top_p>
```

**Example:**
```bash
python generate.py "Hello" 20 0.7 0.9
```

---

## 📝 Generation Parameters

| Parameter     | Description                  | Example         |
| ------------- | ---------------------------- | --------------- |
| `input_text`  | Starting text for generation | `"Hello world"` |
| `num_tokens`  | Number of tokens to generate | `50`            |
| `temperature` | Randomness control (0.0-2.0) | `0.7`           |
| `top_p`       | Nucleus sampling threshold   | `0.9`           |

---

## 🏗️ Architecture Highlights

- **🔥 DeepSeek Multi-Head Latent Attention** - Enhanced attention mechanism
- **⚖️ Mixture-of-Experts** - Scalable expert routing
- **⚡ PyTorch DDP** - Efficient multi-GPU orchestration
- **🎯 Mixed Precision** - FP16/BF16 training optimization

---