

__all__ = ["EarlyExitMgr"]

import torch.nn as nn
from modules import BranchPoint


class EarlyExitMgr(object):
    def __init__(self):
        self.exit_points = []

    def attach_exits(self, model, exits_def):
        # For each exit point, we:
        # 1. Cache the name of the exit_point module (i.e. the name of the module
        #    whose output we forward to the exit branch).
        # 2. Override the exit_point module with an instance of BranchPoint
        for exit_point, exit_branch in exits_def:
            self.exit_points.append(exit_point)
            replaced_module = _find_module(model, exit_point)
            assert replaced_module is not None, "Could not find exit point {}".format(exit_point)
            parent_name, node_name = _split_module_name(exit_point)
            parent_module = _find_module(model, parent_name)
            # Replace the module `node_name` with an instance of `BranchPoint`
            parent_module.__setattr__(node_name, BranchPoint(replaced_module, exit_branch))

    def get_exits_outputs(self, model):
        """Collect the outputs of all the exits and return them.

        The output of each exit was cached during the network forward.
        """
        outputs = []
        for exit_point in self.exit_points:
            branch_point = _get_branch_point_module(model, exit_point)
            output = branch_point.output
            assert output is not None
            outputs.append(output)
        return outputs

    def delete_exits_outputs(self, model):
        """Delete the outputs of all the exits.

        Some exit branches may not be traversed, so we need to delete the cached
        outputs to make sure these outputs do not participate in the backprop.
        """
        outputs = []
        for exit_point in self.exit_points:
            branch_point = _get_branch_point_module(model, exit_point)
            branch_point.output = None
        return outputs


def _find_module(model, mod_name):
    """Locate a module, given its full name"""
    for name, module in model.named_modules():
        if name == mod_name:
            return module
    return None


def _split_module_name(mod_name):
    name_parts = mod_name.split('.')
    parent = '.'.join(name_parts[:-1])
    node = name_parts[-1]
    return parent, node


def _get_branch_point_module(model, exit_point):
    parent_name, node_name = _split_module_name(exit_point)
    parent_module = _find_module(model, parent_name)
    try:
        branch_point = parent_module.__getattr__(node_name)
    except AttributeError:
        # This handles the case where the parent module was data-paralleled after model creation
        if isinstance(parent_module, nn.DataParallel):
            branch_point = parent_module.module.__getattr__(node_name)
            assert isinstance(branch_point, BranchPoint)
        else:
            raise
    return branch_point
