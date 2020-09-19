
import torch
import torch.nn as nn


class Matmul(nn.Module):
    """
    A wrapper module for matmul operation between 2 tensors.
    """
    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return a.matmul(b)


class BatchMatmul(nn.Module):
    """
    A wrapper module for torch.bmm operation between 2 tensors.
    """
    def __init__(self):
        super(BatchMatmul, self).__init__()

    def forward(self, a: torch.Tensor, b:torch.Tensor):
        return torch.bmm(a, b)