## This code creates a simple Transformer Large Language Model using Pytorch and Lighting ai.
### Some things this code does:
* Can train a BPE-tokenizer from scratch.
* Can tokenize a dataset using a pre-trained BPE-tokenizer.
* Train a transformer model from scratch.
* Generate text from a pre-trained model.
* Use Mixture-of-Experts.
* Use Multi-Head Latent Attention from DeepSeek.
* Use Multi-GPU training using fabric.
* Use Mixed-Precision training.
### Some things this code can't do:
* Load model's from Huggingface or anyother AI hub.
* Fine-tune or train a model using RLHF.
* Load any other tokenization algorithm except BPE.
* Can't use safetensors.
* And there are many more things it can't do, but I will stop now.
### Here is how to use this code after cloning the repo:
1. Install the packages needed by running `pip install -r requirements.txt`.
2. Train tokenizer by running `python train_tokenizer.py`.
3. Tokenize the dataset by running `python tokenize_data.py`.
4. Now train a transformer model by running `python train.py`.
5. Want to checkout how the model is? Run `python generate.py "input_text" "num_tokens_to_generate" "temperature" "top_p"`. For example: `python generate.py "Hello" 20 0.7 0.9`
