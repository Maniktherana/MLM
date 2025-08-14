from __future__ import annotations
import json
import modal
import torch
from transformers import AutoTokenizer

import src.config as cfg
from src.model.transformer import GPT, generate

app = modal.App(cfg.APP_NAME)
vol = modal.Volume.from_name(cfg.VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0+cu121", index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("transformers>=4.42.0")
)


@app.function(image=image, gpu=modal.gpu.L4(), volumes={"/vol": vol}, timeout=600)
def generate_text(prompt: str = "Hello", max_new_tokens: int = 120):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, FINAL_DIR, TOK_DIR = cfg.get_paths()

    print("===> Loading model...")
    cfg_path = FINAL_DIR / "config.json"
    model_cfg = json.loads(cfg_path.read_text())

    tok_path = FINAL_DIR / "tokenizer"
    if not tok_path.exists():
        tok_path = TOK_DIR
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)

    model = GPT(model_cfg)
    state = torch.load(FINAL_DIR / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"===> Generating for: '{prompt}'")
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    x = torch.tensor(
        [ids[-model_cfg["context_length"] :]], dtype=torch.long, device=device
    )

    out = generate(
        model=model,
        idx=x,
        max_new_tokens=max_new_tokens,
        context_size=model_cfg["context_length"],
        top_k=25,
        temperature=1.4,
    )

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(text)
    return text


if __name__ == "__main__":
    print("Run with: modal run src.generate::generate_text --prompt 'Your prompt here'")
