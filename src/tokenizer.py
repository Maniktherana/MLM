from __future__ import annotations
from pathlib import Path
from typing import Iterable
from transformers import AutoTokenizer
from src.config import TOK_DIR, VOCAB_SIZE, TOKENIZER_SAMPLE_FRACTION, MIN_CHARS, SEED
from datasets import Dataset


def _text_iter(
    split: Dataset,
    fraction: float = TOKENIZER_SAMPLE_FRACTION,
    min_chars: int = MIN_CHARS,
    seed: int = SEED,
) -> Iterable[str]:
    ds = split.shuffle(seed=seed)
    take = max(1, int(len(ds) * fraction))
    taken = 0
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) >= min_chars:
            yield t
            taken += 1
            if taken >= take:
                break


def ensure_tokenizer(
    train_split: Dataset,
    vocab_size: int = VOCAB_SIZE,
    sample_fraction: float = TOKENIZER_SAMPLE_FRACTION,
    min_chars: int = MIN_CHARS,
    seed: int = SEED,
):
    """
    Load tokenizer from TOK_DIR if present; otherwise train a new byte-level BPE
    (GPT-2 fast base) with <pad>/<bos>/<eos> and save it to TOK_DIR.
    """
    hf_bpe_path = Path("src/hf_bpe_tokenizer")
    if (hf_bpe_path / "tokenizer.json").exists():
        tok = AutoTokenizer.from_pretrained(hf_bpe_path, use_fast=True)
        print(f"[tokenizer] loaded from {hf_bpe_path}, vocab={tok.vocab_size}")
        return tok

    if (TOK_DIR / "tokenizer.json").exists():
        tok = AutoTokenizer.from_pretrained(TOK_DIR, use_fast=True)
        print(f"[tokenizer] loaded from {TOK_DIR}, vocab={tok.vocab_size}")
        return tok

    print("[tokenizer] training new tokenizer…")
    base = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    tok = base.train_new_from_iterator(
        _text_iter(train_split, sample_fraction, min_chars, seed),
        vocab_size=vocab_size,
        new_special_tokens=["<pad>", "<bos>", "<eos>"],
    )

    tok.pad_token = "<pad>"
    tok.bos_token = "<bos>"
    tok.eos_token = "<eos>"

    tok.add_bos_token = False
    tok.add_eos_token = False

    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(TOK_DIR)
    print(f"[tokenizer] saved to {TOK_DIR}, vocab={tok.vocab_size}")
    return tok
