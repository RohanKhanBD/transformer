import torch
from tokenizer import Tokenizer
from model import TransformerLM
from utils import load, ModelConfig
from config_args import generate_args


def main():
    tokenizer = Tokenizer()
    tokenizer.load()

    file_args = generate_args()
    input_tokens = [tokenizer.encode(file_args.input_text) for _ in range(10)]
    num_tokens_to_generate = file_args.num_tokens_to_generate
    temperature = file_args.temperature
    topp = file_args.top_p
    save_file_name = file_args.save_file_name

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = load(save_file_name, map_location=dev)
    model_config: ModelConfig = load(
        save_file_name, "model_config.pt", False, map_location=dev.type
    )
    model_config.flash = False
    model = TransformerLM(model_config, tokenizer.vocab_size).to(dev)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stop_tokens = [tokenizer.special_token["<|endoftext|>"]]
    generated_tokens = model.generate(
        dev,
        tokenizer.special_token["<pad>"],
        stop_tokens,
        input_tokens,
        num_tokens_to_generate,
        temperature,
        topp,
    )
    for tokens in generated_tokens:
        print(tokenizer.decode(tokens))
        print("-" * 50)


if __name__ == "__main__":
    main()
