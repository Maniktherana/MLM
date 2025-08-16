from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length: int):
        self.ds = dataset_split
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id
        assert self.pad_id is not None, "pad_token_id must be set on the tokenizer"
        assert self.eos_id is not None, "eos_token_id must be set on the tokenizer"
        assert self.bos_id is not None, "bos_token_id must be set on the tokenizer"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]
        T = self.max_length
        need = T + 1

        ids = self.tok.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=need - 2,
        )

        seq = []

        seq.append(self.tok.bos_token_id)
        seq.extend(ids)
        seq.append(self.eos_id)

        # Pad to required length
        if len(seq) < need:
            seq += [self.pad_id] * (need - len(seq))

        x = torch.tensor(seq[:T], dtype=torch.long)
        y = torch.tensor(seq[1 : T + 1], dtype=torch.long)
        return x, y


def create_dataloader(
    dataset_split,
    tokenizer,
    batch_size=8,
    max_length=1024,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    ds = GPTDataset(dataset_split, tokenizer, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
