"""IRRA Byte-BPE只读CPU探针；不加载IRRA模型或导入G。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, importlib.util

TOKENIZER_SHA = "19b06ade7532bcff961c4b90590197fcac0d466f08d0b6b801d3d19a36568d67"
BPE_SHA = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

@dataclass(frozen=True)
class TokenReport:
    content_tokens: tuple[int, ...]
    model_tokens: tuple[int, ...]
    total_length: int
    utf8_length: int
    truncated: bool

class TokenizerProbe:
    def __init__(self, tokenizer, text_length=77):
        if text_length != 77:
            raise ValueError("official IRRA text length is 77")
        self.tokenizer = tokenizer
        self.sot = tokenizer.encoder["<|startoftext|>"]
        self.eot = tokenizer.encoder["<|endoftext|>"]

    def inspect(self, text):
        content = tuple(self.tokenizer.encode(text))
        tokens = [self.sot, *content, self.eot]
        truncated = len(tokens) > 77
        if truncated:
            tokens = tokens[:77]
            tokens[-1] = self.eot
        tokens += [0] * (77-len(tokens))
        return TokenReport(content, tuple(tokens), len(content)+2,
                           len(text.encode("utf-8")), truncated)

    def legal_change(self, before, after):
        a, b = self.inspect(before), self.inspect(after)
        return (not a.truncated and not b.truncated
                and a.model_tokens != b.model_tokens)

    def pair_report(self, before, after):
        a, b = self.inspect(before), self.inspect(after)
        return {"before": asdict(a), "after": asdict(b),
                "raw_changed": before != after,
                "model_tokens_changed": a.model_tokens != b.model_tokens,
                "content_length_delta": len(b.content_tokens)-len(a.content_tokens),
                "new_truncation": b.truncated and not a.truncated}

def official_probe(root=Path("/home/lzf/Attack/IRRA-main/IRRA-main")):
    root = Path(root)
    source = root/"utils/simple_tokenizer.py"
    bpe = root/"data/bpe_simple_vocab_16e6.txt.gz"
    for path, expected in ((source, TOKENIZER_SHA), (bpe, BPE_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("official tokenizer/BPE SHA mismatch")
    spec = importlib.util.spec_from_file_location("h_v1_official_tokenizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return TokenizerProbe(module.SimpleTokenizer(bpe_path=str(bpe)))
