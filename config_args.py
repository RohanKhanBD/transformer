from argparse import ArgumentParser


def tokenize_args(train_tokenizer=False):
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path_huggingface",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="The dataset path from huggingface.",
    )
    parser.add_argument(
        "--dataset_sub_set",
        type=str,
        default="sample-10BT",
        help="Subset of the dataset.",
    )
    parser.add_argument(
        "--tokenizer_file_name",
        type=str,
        default="tokenizer",
        help="The path of the tokenizer.",
    )

    if train_tokenizer:
        parser.add_argument(
            "--tokenizer_train_shard_size",
            type=int,
            default=5_000_000,
            help="The number of char to train the tokenizer on.",
        )
        parser.add_argument(
            "--trust_remote_code",
            action="store_true",
            help="Flag for letting huggingface run remote code from the dataset.",
        )

    else:
        parser.add_argument(
            "--data_file_name",
            type=str,
            default="encoded_data",
            help="The path of the dataset after pre-processing (tokenizing).",
        )
        parser.add_argument(
            "--encoded_dataset_shard_size",
            type=int,
            default=int(1e8),
            help="each shard size of the pre-precessed dataset.",
        )
        parser.add_argument(
            "--load_mistral_tokenizer",
            action="store_true",
            help="Flag for using mistral nemo tokenizer.",
        )

    args = parser.parse_args()
    return args


def sft_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--sft_dataset_path_huggingface",
        type=str,
        default="HuggingFaceTB/smoltalk",
        help="The dataset path from huggingface.",
    )
    parser.add_argument(
        "--sft_dataset_sub_set", type=str, default="all", help="Subset of the dataset."
    )
    parser.add_argument(
        "--tokenizer_file_name",
        type=str,
        default="tokenizer",
        help="The path of the tokenizer.",
    )
    parser.add_argument(
        "--data_file_name",
        type=str,
        default="sft_data",
        help="The path of the dataset after pre-processing (tokenizing).",
    )
    parser.add_argument(
        "--encoded_dataset_shard_size",
        type=int,
        default=int(1e8),
        help="each shard size of the pre-precessed dataset.",
    )
    parser.add_argument(
        "--load_mistral_tokenizer",
        action="store_true",
        help="Flag for using mistral nemo tokenizer.",
    )
    args = parser.parse_args()
    return args


def generate_args():
    parser = ArgumentParser()
    parser.add_argument("--input_text", type=str, help="Input prompt for the model.")
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=100,
        help="Number of tokens for the model to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Controles the randomness of the model.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Selects the probabily above the thress hold.",
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default="lilgpt",
        help="Path to the model and its config.",
    )
    parser.add_argument(
        "--backend", type=str, default="inductor", help="Backend for torch compile."
    )
    parser.add_argument(
        "--tokenizer_file_name",
        type=str,
        default="tokenizer",
        help="The path of the tokenizer.",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Flag for weather or not to use torch compile the model.",
    )
    parser.add_argument(
        "--load_mistral_tokenizer",
        action="store_true",
        help="Flag for using mistral nemo tokenizer.",
    )
    args = parser.parse_args()
    return args


def train_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=18701, help="The total number of steps to train."
    )
    parser.add_argument(
        "--eval_rate",
        type=int,
        default=50,
        help="How many steps after the model should be evaluated.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="The total number of steps to eval."
    )
    parser.add_argument(
        "--save_rate",
        type=int,
        default=100,
        help="How many steps after the model should be saved.",
    )
    parser.add_argument(
        "--warm_up",
        type=int,
        default=715,
        help="How many steps should it take for the learning rate (lr) should warm up to the max value.",
    )
    parser.add_argument(
        "--muon_warm_up",
        type=int,
        default=300,
        help="How many steps should it take for the muon momentum should warm up to the max value.",
    )
    parser.add_argument(
        "--muon_cooldown",
        type=int,
        default=50,
        help="How many steps should it take for the muon momentum should cooldown to the min value.",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=2**19,
        help="The number of tokens that should be precessed in a single step.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="How many small batches or micro batches should the total number of tokens be divided into.",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="The seed for controlling model init."
    )
    parser.add_argument(
        "--promissed_flops",
        type=int,
        default=312e12,
        help="The top flops the training hardware can reach.",
    )
    parser.add_argument(
        "--muon_lr",
        type=float,
        default=0.02,
        help="The learning rate for muon (muon_lr).",
    )
    parser.add_argument(
        "--adamw_lr",
        type=float,
        default=3e-3,
        help="The learning rate for adamw (adamw_lr).",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.95, help="The optimizer momentum"
    )
    parser.add_argument(
        "--min_muon_lr",
        type=float,
        default=0.02 * 0.1,
        help="The minimum muon learning rate the warm up should start from and end on.",
    )
    parser.add_argument(
        "--min_adamw_lr",
        type=float,
        default=3e-3 * 0.1,
        help="The minimum adamw learning rate the warm up should start from and end on.",
    )
    parser.add_argument(
        "--min_momentum",
        type=float,
        default=0.85,
        help="The minimum muon momentum the warm up should start from and end on.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="The weight decay for adamw."
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 for adamw.")
    parser.add_argument(
        "--beta2", type=float, default=0.97, help="The beta2 for adamw."
    )
    parser.add_argument(
        "--backend", type=str, default="inductor", help="Backend for torch compile."
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default="lilgpt",
        help="Path to the model and its config.",
    )
    parser.add_argument(
        "--data_file_name",
        type=str,
        default="encoded_data",
        help="The path of the dataset after pre-processing (tokenizing).",
    )
    parser.add_argument(
        "--tokenizer_file_name",
        type=str,
        default="tokenizer",
        help="The path of the tokenizer.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="The dtype for mix-presition training. (bf16 and f16 are supported)",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Flag for weather or not to use torch compile the model.",
    )
    parser.add_argument(
        "--use_autocast",
        action="store_true",
        help="Flag for weather or not to use autocast for training.",
    )
    parser.add_argument(
        "--load_mistral_tokenizer",
        action="store_true",
        help="Flag for using mistral nemo tokenizer.",
    )
    args = parser.parse_args()
    return args


def sft_train_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=1525, help="The total number of steps to train."
    )
    parser.add_argument(
        "--eval_rate",
        type=int,
        default=50,
        help="How many steps after the model should be evaluated.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="The total number of steps to eval."
    )
    parser.add_argument(
        "--save_rate",
        type=int,
        default=100,
        help="How many steps after the model should be saved.",
    )
    parser.add_argument(
        "--warm_up",
        type=int,
        default=715,
        help="How many steps should it take for the learning rate (lr) should warm up to the max value.",
    )
    parser.add_argument(
        "--muon_warm_up",
        type=int,
        default=300,
        help="How many steps should it take for the muon momentum should warm up to the max value.",
    )
    parser.add_argument(
        "--muon_cooldown",
        type=int,
        default=50,
        help="How many steps should it take for the muon momentum should cooldown to the min value.",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=2**19,
        help="The number of tokens that should be precessed in a single step.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="How many small batches or micro batches should the total number of tokens be divided into.",
    )
    parser.add_argument(
        "--promissed_flops",
        type=int,
        default=312e12,
        help="The top flops the training hardware can reach.",
    )
    parser.add_argument(
        "--muon_lr",
        type=float,
        default=0.02,
        help="The learning rate for muon (muon_lr).",
    )
    parser.add_argument(
        "--adamw_lr",
        type=float,
        default=3e-3,
        help="The learning rate for adamw (adamw_lr).",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.95, help="The optimizer momentum"
    )
    parser.add_argument(
        "--min_muon_lr",
        type=float,
        default=0.02 * 0.1,
        help="The minimum muon learning rate the warm up should start from and end on.",
    )
    parser.add_argument(
        "--min_adamw_lr",
        type=float,
        default=3e-3 * 0.1,
        help="The minimum adamw learning rate the warm up should start from and end on.",
    )
    parser.add_argument(
        "--min_momentum",
        type=float,
        default=0.85,
        help="The minimum muon momentum the warm up should start from and end on.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="The weight decay for adamw."
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 for adamw.")
    parser.add_argument(
        "--beta2", type=float, default=0.97, help="The beta2 for adamw."
    )
    parser.add_argument(
        "--backend", type=str, default="inductor", help="Backend for torch compile."
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default="lilgpt",
        help="Path to the model and its config.",
    )
    parser.add_argument(
        "--data_file_name",
        type=str,
        default="sft_data",
        help="The path of the dataset after pre-processing (tokenizing).",
    )
    parser.add_argument(
        "--tokenizer_file_name",
        type=str,
        default="tokenizer",
        help="The path of the tokenizer.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="The dtype for mix-presition training. (bf16 and f16 are supported)",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Flag for weather or not to use torch compile the model.",
    )
    parser.add_argument(
        "--use_autocast",
        action="store_true",
        help="Flag for weather or not to use autocast for training.",
    )
    parser.add_argument(
        "--load_mistral_tokenizer",
        action="store_true",
        help="Flag for using mistral nemo tokenizer.",
    )
    args = parser.parse_args()
    return args
