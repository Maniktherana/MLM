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


def calc_loss_batch(input_batch, target_batch, model, device, tokenizer):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
        ignore_index=tokenizer.pad_token_id,
    )
    return loss


def calc_loss_loader(data_loader, model, device, tokenizer, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device, tokenizer)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, tokenizer, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, tokenizer, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, tokenizer, num_batches=eval_iter
        )
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
        eos_id=tokenizer.eos_token_id,
    )
    
    decoded_text = token_ids_to_text(token_ids, tokenizer, skip_special_tokens=False)
    print(decoded_text.replace("\n", " "))
    debug_text = token_ids_to_text(token_ids, tokenizer, skip_special_tokens=False)
    print(f"[debug] with specials: {debug_text[-120:]}")
    
    model.train()


def save_checkpoint(
    model,
    optimizer,
    step,
    tokenizer,
    train_losses=None,
    val_losses=None,
    track_tokens_seen=None,
):
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CKPT_DIR / f"checkpoint_step_{step:06d}.pt"
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {**cfg.GPT_CONFIG_124M, "vocab_size": tokenizer.vocab_size},
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"===> Checkpoint saved: {checkpoint_path}")

    if (
        train_losses is not None
        or val_losses is not None
        or track_tokens_seen is not None
    ):
        data_path = CKPT_DIR / f"checkpoint_step_{step:06d}.json"
        data = {
            "step": step,
            "train_losses": train_losses or [],
            "val_losses": val_losses or [],
            "tokens_seen": track_tokens_seen or [],
            "timestamp": time.time(),
        }
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"===> Training data saved: {data_path}")

    checkpoints = list(CKPT_DIR.glob("checkpoint_step_*.pt"))
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[2]))

    if len(checkpoints) > 3:
        for old_ckpt in checkpoints[:-3]:
            step_num = old_ckpt.stem.split("_")[2]
            json_file = CKPT_DIR / f"checkpoint_step_{step_num}.json"

            old_ckpt.unlink()
            if json_file.exists():
                json_file.unlink()
            print(f"===> Removed old checkpoint: step_{step_num}")

    json_files = list(CKPT_DIR.glob("checkpoint_step_*.json"))
    for json_file in json_files:
        step_num = json_file.stem.split("_")[2]
        pt_file = CKPT_DIR / f"checkpoint_step_{step_num}.pt"
        if not pt_file.exists():
            json_file.unlink()
            print(f"===> Cleaned up orphaned JSON: step_{step_num}")

    vol.commit()
    print("===> Checkpoint committed to volume")


def load_latest_checkpoint(model, optimizer, tokenizer):
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()

    if not CKPT_DIR.exists():
        print("===> No checkpoint directory found, starting from scratch")
        return 0, [], [], []

    checkpoints = sorted(CKPT_DIR.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        print("===> No checkpoints found, starting from scratch")
        return 0, [], [], []

    latest_checkpoint = checkpoints[-1]
    print(f"===> Loading checkpoint: {latest_checkpoint}")

    # Load .pt file
    checkpoint = torch.load(latest_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.tie_weights()
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_step = checkpoint["step"]

    # Load .json file with training data
    step_num = latest_checkpoint.stem.split("_")[2]
    data_path = CKPT_DIR / f"checkpoint_step_{step_num}.json"

    train_losses, val_losses, track_tokens_seen = [], [], []

    if data_path.exists():
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
            train_losses = data.get("train_losses", [])
            val_losses = data.get("val_losses", [])
            track_tokens_seen = data.get("tokens_seen", [])
            print(f"===> Loaded training data: {len(train_losses)} loss records")
        except Exception as e:
            print(f"===> Warning: Could not load training data: {e}")
    else:
        print("===> No training data found, starting fresh")

    print(f"===> Resumed from step {start_step}")
    if track_tokens_seen:
        print(f"===> Total tokens processed: {track_tokens_seen[-1]:,}")

    return start_step, train_losses, val_losses, track_tokens_seen


def load_training_progress():
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()

    if not CKPT_DIR.exists():
        return None

    progress_files = sorted(CKPT_DIR.glob("training_progress_step_*.json"))
    if not progress_files:
        return None

    latest_progress = progress_files[-1]
    print(f"===> Found training progress file: {latest_progress}")

    try:
        progress_data = json.loads(latest_progress.read_text())
        print(f"===> Loaded progress up to step {progress_data['step']}")
        print(f"===> Found {len(progress_data.get('train_losses', []))} loss records")
        return progress_data
    except Exception as e:
        print(f"===> Error loading progress file: {e}")
        return None


def train_model(model, train_loader, val_loader, optimizer, device, tokenizer):
    total_steps = len(train_loader)

    eval_freq = cfg.TRAINING_CONFIG["eval_freq"]
    eval_iter = cfg.TRAINING_CONFIG["eval_iter"]
    generate_freq = cfg.TRAINING_CONFIG["generate_freq"]

    scaler = GradScaler()

    start_step, train_losses, val_losses, track_tokens_seen = load_latest_checkpoint(
        model, optimizer, tokenizer
    )

    tokens_seen = start_step * cfg.BATCH_SIZE * cfg.MAX_LENGTH
    global_step = start_step

    print(f"===> Training for {total_steps:,} steps (1 complete epoch)")
    print(f"===> Starting from step {start_step}")
    print(
        f"===> Estimated tokens to process: {(total_steps - start_step) * cfg.BATCH_SIZE * cfg.MAX_LENGTH:,}"
    )

    start_time = time.time()
    model.train()

    train_iter = iter(train_loader)

    while global_step < total_steps:
        try:
            input_batch, target_batch = next(train_iter)
        except StopIteration:
            break

        # Skip batches if resuming from checkpoint
        if global_step < start_step:
            global_step += 1
            continue

        optimizer.zero_grad()

        with autocast(device):
            loss = calc_loss_batch(input_batch, target_batch, model, device, tokenizer)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        tokens_seen += input_batch.numel()
        global_step += 1

        if global_step % eval_freq == 0:
            tr, va = evaluate_model(
                model, train_loader, val_loader, device, tokenizer, eval_iter
            )
            train_losses.append(tr)
            val_losses.append(va)
            track_tokens_seen.append(tokens_seen)

            elapsed_mins = (time.time() - start_time) / 60
            progress_pct = (global_step / total_steps) * 100

            print(
                f"  Step {global_step:06d} ({progress_pct:.1f}%): "
                f"Train {tr:.3f}, Val {va:.3f} | {elapsed_mins:.1f}m"
            )

            save_checkpoint(
                model,
                optimizer,
                global_step,
                tokenizer,
                train_losses,
                val_losses,
                track_tokens_seen,
            )

        if global_step % generate_freq == 0:
            prompts = cfg.TRAINING_CONFIG["generation_prompts"]
            prompt_idx = (global_step // generate_freq) % len(prompts)
            current_prompt = prompts[prompt_idx]
            generate_and_print_sample(model, tokenizer, device, current_prompt)

    print(f"===> Training completed after {global_step} steps!")
    return train_losses, val_losses, track_tokens_seen


def save_final_model(model, tokenizer, train_losses, val_losses, track_tokens_seen):
    ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = cfg.get_paths()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = {**cfg.GPT_CONFIG_124M, "vocab_size": tokenizer.vocab_size}
    config_path = FINAL_DIR / "config.json"
    config_path.write_text(json.dumps(model_cfg, indent=2))

    model_path = FINAL_DIR / "model.pt"
    torch.save(model.state_dict(), model_path)

    tokenizer_path = FINAL_DIR / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)

    # Save training history
    history_path = FINAL_DIR / "training_history.json"
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": track_tokens_seen,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"===> Final model saved to {FINAL_DIR}")
    print(f"===> Config: {config_path}")
    print(f"===> Model: {model_path}")
    print(f"===> Tokenizer: {tokenizer_path}")
    print(f"===> Training history: {history_path}")

    vol.commit()
    print("===> Final model committed to volume")


def sample_dataset(dataset_name, revision, sample_fraction, seed):
    ds = load_dataset(dataset_name, revision=revision)["train"]

    total_size = len(ds)
    sample_size = int(total_size * sample_fraction)
    print(
        f"===> Sampling {sample_size:,} examples ({sample_fraction:.1%}) from {total_size:,} total examples"
    )

    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(seed))[
        :sample_size
    ].tolist()
    raw_subset = ds.select(indices)

    train_val_split = raw_subset.train_test_split(test_size=0.1, seed=seed)
    train_split = train_val_split["train"]
    val_split = train_val_split["test"]

    print(
        f"===> Split into {len(train_split):,} train / {len(val_split):,} val (90/10)"
    )
    print("===> First few examples:")
    for i in range(min(3, len(train_split))):
        text_preview = (
            train_split[i]["text"][:100] + "..."
            if len(train_split[i]["text"]) > 100
            else train_split[i]["text"]
        )
        print(f"  {i + 1}: {text_preview}")

    return train_split, val_split


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, timeout=86400)
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"===> Using device: {device}")

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print("===> Loading dataset...")
    train_split, val_split = sample_dataset(
        cfg.DATASET_NAME, cfg.REVISION, cfg.TRAIN_SPLIT_FRACTION, cfg.SEED
    )

    print("===> Ensuring tokenizer...")
    tokenizer = ensure_tokenizer(train_split, cfg.VOCAB_SIZE)

    print("===> Creating data loaders...")
    train_loader = create_dataloader(
        train_split, tokenizer, cfg.BATCH_SIZE, cfg.MAX_LENGTH, shuffle=True
    )
    val_loader = create_dataloader(
        val_split,
        tokenizer,
        cfg.BATCH_SIZE,
        cfg.MAX_LENGTH,
        shuffle=False,
        drop_last=False,
    )

    print("===> Creating model...")
    model_cfg = {**cfg.GPT_CONFIG_124M, "vocab_size": tokenizer.vocab_size}
    model = GPT(model_cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"===> Total parameters: {total_params:,}")

    print("===> Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    print("===> Starting training...")
    train_losses, val_losses, track_tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device, tokenizer
    )

    print("===> Saving final model...")
    save_final_model(model, tokenizer, train_losses, val_losses, track_tokens_seen)

    print("===> Training complete!")
    return {
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "total_steps": len(train_losses) * cfg.TRAINING_CONFIG["eval_freq"]
        if train_losses
        else 0,
        "total_tokens": track_tokens_seen[-1] if track_tokens_seen else 0,
    }


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
