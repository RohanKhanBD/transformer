# ---------------------------------------
# Ints
# ---------------------------------------
# training params
epochs: int = 5243

# eval params
eval_rate: int = 50
eval_epochs: int = 100

# saving rate
save_rate: int = 100

# learning rate schedule params
warm_up: int = 6

# batch sizes
total_batch_size = 2**16
batch_size: int = 4

# encoded dataset shard size
encoded_dataset_shard_size = int(1e8)

# non encoded dataset shard size
non_encoded_dataset_shard_size = 500

# tokenizer train shard size
tokenizer_train_shard_size = 500

# ---------------------------------------
# Floats
# ---------------------------------------
# learning rates
lr: float = 3e-3
min_lr: float = lr * 0.1

# optimizer params
weight_decay: float = 0.1
betas = (0.9, 0.97)

# ---------------------------------------
# Strings
# ---------------------------------------
# torch.compile params
backend = "inductor"

# save file names
save_fie_name = "nanogpt"
data_file_name = "encoded_data"

# dataset path in huggingface datasets
dataset_path_huggingface = "openwebtext"
dataset_sub_set: str | None = None

# ---------------------------------------
# Bools
# ---------------------------------------
# torch.compile params
compile_model = True

# trusting datasets remote code
trust_remote_code = True
