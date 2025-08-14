from __future__ import annotations
import json
import modal
import torch
import random
import numpy as np
from datasets import load_dataset
import time
from torch.amp import autocast, GradScaler

import src.config as cfg
from src.dataset import create_dataloader
from src.model.transformer import GPT, generate_text_simple
from src.tokenizer import ensure_tokenizer

app = modal.App(cfg.APP_NAME)
vol = modal.Volume.from_name(cfg.VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0+cu121", index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("transformers>=4.42.0", "datasets>=2.20.0", "tokenizers>=0.15.2")
)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def text_to_token_ids(text: str, tokenizer, add_bos: bool = True, device=None):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    t = torch.tensor([ids], dtype=torch.long)
    return t.to(device) if device is not None else t


def token_ids_to_text(token_ids, tokenizer, skip_special_tokens: bool = True):
    if torch.is_tensor(token_ids):
        if token_ids.dim() == 2:
            token_ids = token_ids[0].tolist()
        else:
            token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer, add_bos=True, device=device)
    token_ids = generate_text_simple(
        model=model,
        idx=encoded,
        max_new_tokens=50,
        context_size=context_size,
    )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def load_latest_checkpoint(model, optimizer, tokenizer):
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()
    
    if not CKPT_DIR.exists():
        print("===> No checkpoint directory found, starting from scratch")
        return 0
    
    checkpoints = sorted(CKPT_DIR.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        print("===> No checkpoints found, starting from scratch")
        return 0
    
    latest_checkpoint = checkpoints[-1]
    print(f"===> Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"]
    
    print(f"===> Resumed from step {start_step}")
    return start_step


def train_model(model, train_loader, val_loader, optimizer, device, tokenizer):
    num_epochs = 3
    eval_freq = 200
    eval_iter = 10
    start_context = "Hello"
    checkpoint_freq = 1000

    scaler = GradScaler(device)

    start_step = load_latest_checkpoint(model, optimizer, tokenizer)

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = start_step * cfg.BATCH_SIZE * cfg.MAX_LENGTH
    global_step = start_step

    print(
        f"===> Training for {num_epochs} epochs, {len(train_loader)} batches per epoch"
    )
    print(f"===> Starting from step {start_step}")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Skip batches we've already processed
            current_batch = epoch * len(train_loader) + batch_idx
            if current_batch < start_step:
                continue

            optimizer.zero_grad()

            with autocast(device):
                loss = calc_loss_batch(input_batch, target_batch, model, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                tr, va = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(tr)
                val_losses.append(va)
                track_tokens_seen.append(tokens_seen)

                elapsed_mins = (time.time() - start_time) / 60
                print(
                    f"  Step {global_step:06d}: Train {tr:.3f}, Val {va:.3f} | {elapsed_mins:.1f}m"
                )

            if global_step % checkpoint_freq == 0:
                save_checkpoint(model, optimizer, global_step, tokenizer)

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def save_checkpoint(model, optimizer, step, tokenizer):
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CKPT_DIR / f"checkpoint_step_{step}.pt"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {**cfg.GPT_CONFIG_124M, "vocab_size": tokenizer.vocab_size},
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"===> Checkpoint saved: {checkpoint_path}")

    checkpoints = list(CKPT_DIR.glob("checkpoint_step_*.pt"))
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    if len(checkpoints) > 3:
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()
            print(f"===> Removed old checkpoint: {old_ckpt.name}")

    vol.commit()
    print("===> Checkpoint committed to volume")


def sample_dataset(dataset_name, revision):
    ds = load_dataset(dataset_name, revision=revision)["train"]
    
    total_size = len(ds)
    sample_size = int(total_size * cfg.TOKENIZER_SAMPLE_FRACTION)
    print(f"===> Sampling {sample_size:,} examples ({cfg.TOKENIZER_SAMPLE_FRACTION:.1%}) from {total_size:,} total examples")
    
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(cfg.SEED))[:sample_size].tolist()
    ds = ds.select(indices)
    
    split = ds.train_test_split(test_size=0.01, seed=cfg.SEED)
    print("===> First few examples:")
    for i in range(min(5, len(split["train"]))):
        print(split["train"][i]["text"])
    return split["train"], split["test"]


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, timeout=86400)
def train():
    dataset_name = "Skylion007/openwebtext"
    revision = "refs/convert/parquet"
    lr = 3e-4
    weight_decay = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    
    print("===> Using device:", device)
    print("===> Loading dataset...")
    train_split, val_split = sample_dataset(dataset_name, revision)

    print("===> Ensuring tokenizer...")
    tokenizer = ensure_tokenizer(train_split)
    tokenizer.model_max_length = cfg.GPT_CONFIG_124M["context_length"] + 1

    print("===> Creating dataloaders...")
    train_loader = create_dataloader(
        train_split,
        tokenizer,
        batch_size=cfg.BATCH_SIZE,
        max_length=cfg.GPT_CONFIG_124M["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    print("===> Creating validation dataloader...")
    val_loader = create_dataloader(
        val_split,
        tokenizer,
        batch_size=cfg.BATCH_SIZE,
        max_length=cfg.GPT_CONFIG_124M["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print("===> Creating model...")
    model_cfg = {**cfg.GPT_CONFIG_124M, "vocab_size": tokenizer.vocab_size}
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    print("===> Training model...")
    train_losses, val_losses, track_tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device, tokenizer
    )

    print("===> Saving model...")
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "tokenizer").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), FINAL_DIR / "model.pt")
    (FINAL_DIR / "config.json").write_text(json.dumps(model_cfg, indent=2))
    tokenizer.save_pretrained(FINAL_DIR / "tokenizer")

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": track_tokens_seen,
        "hyperparameters": {
            "epochs": 3,
            "lr": lr,
            "weight_decay": weight_decay,
            "eval_freq": 200,
            "eval_iter": 10,
            "dataset": dataset_name,
            "revision": revision,
        },
    }
    (FINAL_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"===> [done] saved to {FINAL_DIR}")
    print(f"===> Training metrics saved to {FINAL_DIR / 'training_metrics.json'}")

    vol.commit()
    print("===> Volume committed")


@app.function(image=image, volumes={"/vol": vol})
def download_model(local_path: str = "./"):
    import shutil
    from pathlib import Path

    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"===> Downloading model from {FINAL_DIR} to {local_path}")

    if FINAL_DIR.exists():
        for item in FINAL_DIR.iterdir():
            if item.is_file():
                shutil.copy2(item, local_path / item.name)
                print(f"Downloaded: {item.name}")
            elif item.is_dir():
                shutil.copytree(item, local_path / item.name, dirs_exist_ok=True)
                print(f"Downloaded directory: {item.name}")
        print(f"Model downloaded successfully to {local_path}")
        return str(local_path.absolute())
    else:
        print("No trained model found. Run training first!")
        return None


@app.function(image=image, volumes={"/vol": vol})
def list_saved_files():
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()

    print(f"Contents of {FINAL_DIR}:")
    if FINAL_DIR.exists():
        for item in FINAL_DIR.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"===> {item.relative_to(FINAL_DIR)} ({size_mb:.2f} MB)")
    else:
        print("===> No files found. Run training first!")

    return list(FINAL_DIR.rglob("*")) if FINAL_DIR.exists() else []


if __name__ == "__main__":
    print("Run with: modal run src.train::train")
