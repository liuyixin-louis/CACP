
"""Modules related to a model's topology"""
import torch.nn as nn


class BranchPoint(nn.Module):
    """Add a branch to an existing model."""
    def __init__(self, branched_module, branch_net):
        """
        :param branched_module: the module in the original network to which we add a branch.
        :param branch_net: the new branch
        """
        super().__init__()
        self.branched_module = branched_module
        self.branch_net = branch_net
        self.output = None

    def forward(self, x):
        x1 = self.branched_module.forward(x)
        self.output = self.branch_net.forward(x1)
        return x1


# This class is "borrowed" from PyTorch 1.3 until we integrate it
class Flatten(nn.Module):
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


# A temporary trick to see if we need to add Flatten to the `torch.nn` namespace for convenience.
try:
    Flatten = nn.Flatten
except AttributeError:
    nn.Flatten = Flatten