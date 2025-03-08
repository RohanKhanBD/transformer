import torch
from tokenizer import Tokenizer
from model import TransformerLM
from argparse import ArgumentParser
from utils import load, ModelConfig


def main():
    tokenizer = Tokenizer()
    tokenizer.load()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = load("nanogpt", map_location=dev)
    model_config: ModelConfig = load(
        "nanogpt", False, "model_config.pt", map_location=dev.type
    )
    model_config.flash = False
    model = TransformerLM(model_config, tokenizer.vocab_size).to(dev)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    parser = ArgumentParser()
    parser.add_argument("input_text", type=str)
    parser.add_argument("num_tokens_to_generate", type=int)
    parser.add_argument("temperature", type=float)
    parser.add_argument("top_p", type=float)

    args = parser.parse_args()

    input_tokens = [tokenizer.encode(args.input_text) for _ in range(10)]
    num_tokens_to_generate = args.num_tokens_to_generate
    temperature = args.temperature
    topp = args.top_p

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
