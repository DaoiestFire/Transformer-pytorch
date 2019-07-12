# coding = utf-8
from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import os
import linecache

class CoupletDataset(Dataset):
    """Couplet Dataset"""
    def __init__(self, x_file, y_file):
        assert os.path.exists(x_file) and os.path.exists(y_file), "the file path is not exists."
        self.x = linecache.getlines(x_file)
        self.y = linecache.getlines(y_file)
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

couplet_dataset = CoupletDataset(r"data\train\in.txt", r"data\train\out.txt")

couplet_dataloader = DataLoader(couplet_dataset, batch_size=4,
                        shuffle=True,num_workers=0)

for i_batch, sample_batched in enumerate(couplet_dataloader):
    x, y = sample_batched
    print(x, y)
