import torch
from torch import nn

from src.model.attention import MultiHeadAttention
from src.model.model_utils import FeedForward, LayerNorm


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        self.tie_weights()

    def tie_weights(self):
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def _apply_repetition_penalty(
    logits: torch.Tensor, sequences: torch.Tensor, penalty: float
):
    """Decrease logits of tokens that already appeared in each row."""
    if penalty == 1.0:
        return logits
    B, V = logits.shape
    for b in range(B):
        seen = torch.bincount(sequences[b].to(torch.long), minlength=V).bool()
        logits[b, seen] /= penalty
    return logits


def _top_k_filter(logits: torch.Tensor, top_k: int):
    """Keep only top_k tokens per row, mask the rest to -inf."""
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)  # (B,1)
    return logits.masked_fill(logits < kth, float("-inf"))


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = 50,
    eos_id: int | None = None,
    repetition_penalty: float = 1.0,
):
    device = idx.device
    B = idx.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        logits = model(idx_cond)[:, -1, :]

        if eos_id is not None:
            logits[finished] = float("-inf")
            logits[finished, eos_id] = 0.0

        if repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, idx, repetition_penalty)

        if temperature > 0:
            logits = logits / float(temperature)

        logits = _top_k_filter(logits, top_k)

        probs = torch.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)\

        if eos_id is not None:
            finished |= next_ids.squeeze(1) == eos_id

        idx = torch.cat([idx, next_ids], dim=1)

        if eos_id is not None and torch.all(finished):
            break

    return idx


def generate_text_simple(model, idx, max_new_tokens, context_size, eos_id):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        if idx.size(0) == 1:
            if next_id.item() == eos_id:
                break
    return idx
