import regex
from tqdm import tqdm
from utils import load, save
from collections import defaultdict


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_states(ids):
    counts = defaultdict(int)
    for word, freq in ids.items():
        symb = word.split()
        for s in range(len(symb) - 1):
            counts[(symb[s], symb[s + 1])] += freq
    return counts


def get_pairs(ids):
    pairs = set()
    prev = ids[0]
    for char in ids[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs


def merge(ids: list, pair: tuple):
    new_id = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair == (ids[i], ids[i + 1]):
            new_id.append(pair[0] + pair[1])
            i += 2
        else:
            new_id.append(ids[i])
            i += 1
    return new_id


class Tokenizer:
    def __init__(self, vocab_size=0):
        self.pattern = regex.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        self.vocab_size = vocab_size
        self.byte_encode = bytes_to_unicode()
        self.byte_decode = {v: k for k, v in self.byte_encode.items()}
        self.vocab = self.byte_decode.copy()
        self.vocab_decode = self.byte_encode.copy()
        self.merges = {}
        self.special_token = {}
        self.invers_special_token = {}
        self.catch = {}

    def utf_encode_text(self, text: str):
        return "".join(self.byte_encode[b] for b in text.encode("utf-8"))

    def utf_decode_text(self, text: str):
        return bytearray([self.byte_decode[c] for c in text]).decode(
            "utf-8", errors="replace"
        )

    def regester_special_token(self, special_token: dict):
        self.special_token = {k: v + len(self.vocab) for k, v in special_token.items()}
        self.invers_special_token = {v: k for k, v in self.special_token.items()}
        self.vocab_size += len(self.special_token)

    def train(self, data: str):
        lenth = len(self.vocab)
        if lenth == self.vocab_size:
            return True
        assert self.vocab_size >= lenth
        num_merges = self.vocab_size - lenth
        text_chunks = regex.findall(self.pattern, data)
        ids = defaultdict(int)
        for chunk in text_chunks:
            b = " ".join(self.utf_encode_text(chunk))
            ids[b] += 1
        i = 0
        while i < num_merges:
            state = get_states(ids)
            if not state:
                break
            pair = max(state, key=state.get)
            new_ids = defaultdict(int)
            for k, v in ids.items():
                new_ids[" ".join(merge(k.split(), pair))] = v
            ids = new_ids.copy()
            idx = lenth + i
            if pair[0] + pair[1] in self.vocab:
                continue
            self.vocab[pair[0] + pair[1]] = idx
            self.vocab_decode[idx] = pair[0] + pair[1]
            self.merges[pair] = idx
            print(f"index:{i + 1}/{num_merges} | pair:{pair} => {pair[0] + pair[1]}")
            i += 1
        if len(self.vocab) == self.vocab_size:
            return True
        return False

    def bpe(self, text_: str):
        if text_ in self.catch:
            return self.catch[text_]
        text = list(text_)
        while True:
            pairs = get_pairs(text)
            if not pairs:
                break
            min_pair = min(pairs, key=lambda pair: self.merges.get(pair, float("inf")))
            if min_pair not in self.merges:
                break
            text = merge(text, min_pair)
            if len(text) == 1:
                break
        self.catch[text_] = " ".join(text)
        return " ".join(text)

    def encode_vocab(self, text: str):
        text_chuncks = regex.findall(self.pattern, text)
        ids = []
        for chunk in text_chuncks:
            b = self.bpe(self.utf_encode_text(chunk))
            ids.extend([self.vocab[s] for s in b.split()])
        return ids

    def encode(self, text: str):
        special_pattern = (
            "(" + "|".join(regex.escape(s) for s in self.special_token) + ")"
        )
        chunks = regex.split(special_pattern, text)
        ids = []
        for chunk in chunks:
            if chunk in self.special_token:
                ids.append(self.special_token[chunk])
            else:
                ids.extend(self.encode_vocab(chunk))
        return ids

    def decode(self, ids: list, disable: bool = False):
        decoded_ids = []
        for i in tqdm(ids, "decoding", disable=disable):
            if i in self.vocab_decode:
                decoded_ids.append(self.vocab_decode[i])
            elif i in self.invers_special_token:
                decoded_ids.append(self.utf_encode_text(self.invers_special_token[i]))
        return self.utf_decode_text("".join(decoded_ids))

    def save(self, file: str):
        save(
            [
                self.vocab_size,
                self.vocab,
                self.vocab_decode,
                self.merges,
                self.special_token,
                self.invers_special_token,
            ],
            file,
            "tokenizer.pt",
        )

    def load(self, file: str):
        (
            self.vocab_size,
            self.vocab,
            self.vocab_decode,
            self.merges,
            self.special_token,
            self.invers_special_token,
        ) = load(file, "tokenizer.pt", False)
