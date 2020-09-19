
import torch
import torch.nn as nn


class Concat(nn.Module):
    def __init__(self, dim=0):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *seq):
        return torch.cat(seq, dim=self.dim)


class Chunk(nn.Module):
    def __init__(self, chunks, dim=0):
        super(Chunk, self).__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, tensor):
        return tensor.chunk(self.chunks, dim=self.dim)


class Split(nn.Module):
    def __init__(self, split_size_or_sections, dim=0):
        super(Split, self).__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, tensor):
        return torch.split(tensor, self.split_size_or_sections, dim=self.dim)


class Stack(nn.Module):
    def __init__(self, dim=0):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, seq):
        return torch.stack(seq, dim=self.dim)
