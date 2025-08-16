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


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, timeout=600)
def generate_text(prompt: str = "Hello"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, FINAL_DIR, TOK_DIR = cfg.get_paths()

    print("===> Loading model...")
    model_cfg = json.loads((FINAL_DIR / "config.json").read_text())

    tok_path = FINAL_DIR / "tokenizer"
    if not tok_path.exists():
        tok_path = TOK_DIR
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)

    model = GPT(model_cfg)
    state = torch.load(FINAL_DIR / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.tie_weights()
    model.to(device).eval()
    torch.set_grad_enabled(False)

    print(f"===> Generating for: '{prompt}'")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    ctx = model_cfg["context_length"]
    x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)

    out = generate(
        model=model,
        idx=x,
        max_new_tokens=200,
        context_size=ctx,
        temperature=0.8,
        top_k=50,
        eos_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
    print(text)
    return text


if __name__ == "__main__":
    print("Run with: modal run src.generate::generate_text --prompt 'Your prompt here'")
