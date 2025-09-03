from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainConfig:
    # Training params
    steps: int = 18701
    eval_rate: int = 50
    eval_steps: int = 100
    save_rate: int = 10
    warm_up: int = 715
    seed: int = 1337

    # Batch sizes
    total_batch_size: int = 2**19
    batch_size: int = 4
    
    # Learning rates
    lr: float = 3e-3
    min_lr: float = 3e-4  # lr * 0.1

    # Optimizer params
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.97)

    # Torch compile params
    compile_model: bool = True
    backend: str = "inductor"

    # File names
    save_file_name: str = "nanogpt"


@dataclass
class DataConfig:
    data_file_name: str = "encoded_data"

    # Dataset path in huggingface datasets
    dataset_path_huggingface: str = "HuggingFaceFW/fineweb-edu"
    dataset_sub_set: Optional[str] = "sample-10BT"

    # Dataset shard sizes
    encoded_dataset_shard_size: int = int(1e8)
    tokenizer_train_shard_size: int = 5_000_000

    # Trusting datasets remote code
    trust_remote_code: bool = False
