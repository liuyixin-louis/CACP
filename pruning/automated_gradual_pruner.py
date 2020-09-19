
from .ranked_structures_pruner import *
# from . import create_mask_level_criterion
import pruning


class AgpPruningRate(object):
    """A pruning-rate scheduler per https://arxiv.org/pdf/1710.01878.pdf.
    """
    def __init__(self, initial_sparsity, final_sparsity):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        assert final_sparsity > initial_sparsity

    def step(self, current_epoch, starting_epoch, ending_epoch, freq):
        span = ((ending_epoch - starting_epoch - 1) // freq) * freq
        assert span > 0

        target_sparsity = (self.final_sparsity +
                           (self.initial_sparsity-self.final_sparsity) *
                           (1.0 - ((current_epoch-starting_epoch)/span))**3)

        return target_sparsity


class AutomatedGradualPrunerBase(object):
    """Prune to an exact sparsity level specification using a prescribed sparsity
    level schedule formula.

    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.

    Michael Zhu and Suyog Gupta, "To prune, or not to prune: exploring the
    efficacy of pruning for model compression", 2017 NIPS Workshop on Machine
    Learning of Phones and other Consumer Devices,
    (https://arxiv.org/pdf/1710.01878.pdf)
    """

    def __init__(self, name, rate_scheduler):
        """
        Args:
            rate_scheduler - schedules the pruning rate.  You can plug in any
            rate scheduler.  We implemented AGP paper's rate control in AgpPruningRate.
        """
        self.name = name
        self.agp_pr = rate_scheduler

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        target_sparsity = self.agp_pr.step(meta['current_epoch'], meta['starting_epoch'],
                                           meta['ending_epoch'], meta['frequency'])
        self._set_param_mask_by_sparsity_target(param, param_name, zeros_mask_dict, target_sparsity, meta['model'])

    def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model=None):
        """Set the parameter mask using a target sparsity. Override this in subclasses"""
        raise NotImplementedError


def create_mask_level_criterion(tensor, desired_sparsity):
    """Create a tensor mask using a level criterion.

    A specified fraction of the input tensor will be masked.  The tensor coefficients
    are first sorted by their L1-norm (absolute value), and then the lower `desired_sparsity`
    coefficients are masked.
    Granularity: Element-wise

    WARNING: due to the implementation details (speed over correctness), this will perform
    incorrectly if "too many" of the coefficients have the same value. E.g. this will fail:
        a = torch.ones(3, 64, 32, 32)
        mask = cacp.create_mask_level_criterion(a, desired_sparsity=0.3)
        assert math.isclose(cacp.sparsity(mask), 0.3)

    Args:
        tensor - the tensor to mask.
        desired_sparsity - a floating-point value in the range (0..1) specifying what
            percentage of the tensor will be masked.
    Returns:
        boolean mask tensor, having the same size as the input tensor.
    """
    with torch.no_grad():
        # partial sort
        bottomk, _ = torch.topk(tensor.abs().view(-1),
                                int(desired_sparsity * tensor.numel()),
                                largest=False,
                                sorted=True)
        threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
        mask = create_mask_threshold_criterion(tensor, threshold)
        return mask

class AutomatedGradualPruner(AutomatedGradualPrunerBase):
    """Fine-grained pruning with an AGP sparsity schedule.

    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.
    """
    def __init__(self, name, initial_sparsity, final_sparsity, weights, rate_scheduler_factory=AgpPruningRate):
        super().__init__(name, rate_scheduler=rate_scheduler_factory(initial_sparsity, final_sparsity))
        self.params_names = weights
        assert self.params_names

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.params_names:
            return
        super().set_param_mask(param, param_name, zeros_mask_dict, meta)

    def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model=None):
        zeros_mask_dict[param_name].mask = create_mask_level_criterion(param, target_sparsity)


class StructuredAGP(AutomatedGradualPrunerBase):
    """Structured pruning with an AGP sparsity schedule.

    This is a base-class for structured pruning with an AGP schedule.  It is an
    extension of the AGP concept introduced by Zhu et. al.
    """
    def __init__(self, name, initial_sparsity, final_sparsity, rate_scheduler_factory=AgpPruningRate):
        super().__init__(name, rate_scheduler=rate_scheduler_factory(initial_sparsity, final_sparsity))
        self.pruner = None

    def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model):
        self.pruner._set_param_mask_by_sparsity_target(param, param_name, zeros_mask_dict, target_sparsity, model)


class L1RankedStructureParameterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None, kwargs=None):
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = L1RankedStructureParameterPruner(name, group_type, desired_sparsity=0, weights=weights,
                                                       group_dependency=group_dependency, kwargs=kwargs)


class L2RankedStructureParameterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None, kwargs=None):
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = L2RankedStructureParameterPruner(name, group_type, desired_sparsity=0, weights=weights,
                                                       group_dependency=group_dependency, kwargs=kwargs)


class ActivationAPoZRankedFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = ActivationAPoZRankedFilterPruner(name, group_type, desired_sparsity=0,
                                                       weights=weights, group_dependency=group_dependency)


class ActivationMeanRankedFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = ActivationMeanRankedFilterPruner(name, group_type, desired_sparsity=0,
                                                       weights=weights, group_dependency=group_dependency)

class GradientRankedFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = GradientRankedFilterPruner(name, group_type, desired_sparsity=0,
                                                 weights=weights, group_dependency=group_dependency)


class RandomRankedFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = RandomRankedFilterPruner(name, group_type, desired_sparsity=0,
                                               weights=weights, group_dependency=group_dependency)


class BernoulliFilterPruner_AGP(StructuredAGP):
    def __init__(self, name, initial_sparsity, final_sparsity, group_type, weights, group_dependency=None):
        assert group_type in ['3D', 'Filters']
        super().__init__(name, initial_sparsity, final_sparsity)
        self.pruner = BernoulliFilterPruner(name, group_type, desired_sparsity=0,
                                            weights=weights, group_dependency=group_dependency)
