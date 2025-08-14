from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length: int):
        self.ds = dataset_split
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        assert self.pad_id is not None, "pad_token_id must be set on the tokenizer"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]
        need = self.max_length + 1
        ids = self.tok.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=need,  # avoids long-seq warnings
        )
        if len(ids) < need:
            ids += [self.pad_id] * (need - len(ids))
        x = torch.tensor(ids[: self.max_length], dtype=torch.long)
        y = torch.tensor(ids[1 : self.max_length + 1], dtype=torch.long)
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
