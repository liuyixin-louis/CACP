
import logging
import torch

__all__ = ["find_dependencies"]
msglogger = logging.getLogger()


def find_dependencies(dependency_type, sgraph, layers, layer_name, dependencies_list):
    """Produce a list of pruning-dependent layers.

    Given a SummaryGraph instance (sgraph), this function returns the list of dependent
    layers in `dependencies_list`.
    A "dependent layer" is a layer affected by a change in the number of filters/channels
    (dependency_type) in `layer_name`.
    """
    if dependency_type == "channels":
        return _find_dependencies_channels(sgraph, layers, layer_name, dependencies_list)
    if dependency_type == "filters":
        return _find_dependencies_filters(sgraph, layers, layer_name, dependencies_list)
    raise ValueError("%s is not a valid dependency type" % dependency_type)


def _find_dependencies_channels(sgraph, layers, layer_name, dependencies_list):
    # Find all instances of Convolution layers that immediately precede this layer
    predecessors = sgraph.predecessors_f(layer_name, ['Conv'])
    for predecessor in predecessors:
        dependencies_list.append(predecessor)
        prev = sgraph.find_op(predecessor)

        if prev['attrs']['group'] == prev['attrs']['n_ifm']:
            # This is a group-wise convolution, and a special one at that (groups == in_channels).
            _find_dependencies_channels(sgraph, layers, predecessor, dependencies_list)
        elif prev['attrs']['group'] != 1:
            raise ValueError("CACP AutoCompression currently does not "
                             "handle this convolution groups configuration {} "
                             "(layer={} {}\n{})".format(
                             prev['attrs']['group'], predecessor, prev, layers[predecessor]))


# todo: remove the 'layers' parameter
def _find_dependencies_filters(sgraph, layers, layer_name, dependencies_list):
    # Find all instances of Convolution or FC (GEMM) layers that immediately follow this layer
    successors = sgraph.successors_f(layer_name, ['Conv', 'Gemm'])    
    for successor in successors:
        dependencies_list.append(successor)
        next = sgraph.find_op(successor)
        if next['type'] == 'Conv':
            if next['attrs']['group'] == next['attrs']['n_ifm']:
                # This is a group-wise convolution, and a special one at that (groups == in_channels).
                _find_dependencies_filters(sgraph, layers, successor, dependencies_list)
            elif next['attrs']['group'] != 1:
                raise ValueError("CACP AutoCompression currently does not handle this conv.groups configuration")    
 