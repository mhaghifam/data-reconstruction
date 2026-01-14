import numpy as np
import torch
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class DataGeneration:
    def __init__(self, N, d, delta):
        self.delta = delta
        self.N = N
        self.dim = d
        self.centers = torch.bernoulli(torch.full((self.N, self.dim), 0.5))
        self.train_cluster_ids = None


    def generate_samples(self, n, pad_value=-1):
        cluster_ids = torch.randint(low=0, high=self.N, size=(n,))
        X_clean = self.centers[cluster_ids, :]
        noise = torch.bernoulli(torch.full((n, self.dim), self.delta / 2))
        X_noised = torch.bitwise_xor(X_clean.long(), noise.long())

        # Fix: low=2 ensures input_length >= 1
        lengths = torch.randint(low=2, high=self.dim + 1, size=(n,))
        self.train_cluster_ids = cluster_ids

        col_indices = torch.arange(self.dim)
        mask = col_indices >= lengths.unsqueeze(1)
        X_noised[mask] = pad_value

        unique_element, counts = torch.unique(cluster_ids, return_counts=True)
        singleton = unique_element[counts == 1]
        return X_noised, lengths, singleton

    def generate_fixed_length_samples(self, n, cluster_idx, prefix_len, pad_value=-1):
        """Generate n samples from a specific cluster with fixed prefix length."""
        cluster_ids = cluster_idx * torch.ones(size=(n,), dtype=torch.long)
        X_clean = self.centers[cluster_ids, :]
        noise = torch.bernoulli(torch.full((n, self.dim), self.delta / 2))
        X_noised = torch.bitwise_xor(X_clean.long(), noise.long())

        # Label is at position prefix_len
        Y = X_noised[torch.arange(n), prefix_len]

        # Mask positions >= prefix_len
        col_indices = torch.arange(self.dim)
        mask = col_indices >= prefix_len
        X_noised[:, mask] = pad_value

        return X_noised, Y
    

class NextTokenDataset(Dataset):
    def __init__(self, X, lengths, pad_value=-1):
        """
        X: (n, dim) - padded sequences
        lengths: (n,) - actual length of each sequence
        pad_value: value used for padding
        """
        self.X = X
        self.lengths = lengths
        self.pad_value = pad_value
        self.dim = X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx]
        length = self.lengths[idx].item()  # <-- add .item() to convert to int
        input_seq = seq[:-1]
        target_seq = seq[1:]
        input_length = length - 1
        attn_mask = torch.arange(self.dim - 1) < input_length
        loss_mask = attn_mask.float()  # <-- convert to float explicitly

        return {
            'input': input_seq,
            'target': target_seq,
            'attn_mask': attn_mask.float(),
            'loss_mask': loss_mask,
            'length': input_length
        }