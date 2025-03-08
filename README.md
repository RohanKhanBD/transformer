## This code creats a simple Transformer Large Language Model using Pytorch and Lighting ai.
### Some things this code does:
* Can train a BPE-tokenizer from scratch.
* Can tokenize a dataset using a pre-trained BPE-tokenizer.
* Train a transformer model from scratch.
* Generate text from a pre-trained model.
### Some things this code can't do:
* Load model's from Huggingface or anyother AI hub.
* Fine-tune or train a model using RLHF.
* Load any other tokenization algorithm except BPE.
* Can't use safetensors.
* And there are many more things it can't do, but I will stop now.
### The code structure:
```
configuration.py
generate.py
model.py
README.md
tokenize_data.py
tokenizer.py
train_tokenizer.py
train.py
utils.py
```
