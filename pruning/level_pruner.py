
import pruning


class SparsityLevelParameterPruner(object):
    """Prune to an exact pruning level specification.

    This pruner is very similar to MagnitudeParameterPruner, but instead of
    specifying an absolute threshold for pruning, you specify a target sparsity
    level (expressed as a fraction: 0.5 means 50% sparsity.)

    To find the correct threshold, we view the tensor as one large 1D vector, sort
    it using the absolute values of the elements, and then take topk elements.
    """

    def __init__(self, name, levels, **kwargs):
        self.name = name
        self.levels = levels
        assert self.levels

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        # If there is a specific sparsity level specified for this module, then
        # use it.  Otherwise try to use the default level ('*').
        desired_sparsity = self.levels.get(param_name, self.levels.get('*', 0))
        if desired_sparsity == 0:
            return
        zeros_mask_dict[param_name].mask = pruning.create_mask_level_criterion(param, desired_sparsity)
