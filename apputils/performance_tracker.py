

"""Performance trackers used to track the best performing epochs when training.
"""
import operator
import utils


__all__ = ["TrainingPerformanceTracker",
           "SparsityAccuracyTracker"]


class TrainingPerformanceTracker(object):
    """Base class for performance trackers using Top1 and Top5 accuracy metrics"""
    def __init__(self, num_best_scores):
        self.perf_scores_history = []
        self.max_len = num_best_scores

    def reset(self):
        self.perf_scores_history = []

    def step(self, model, epoch, **kwargs):
        """Update the list of top training scores achieved so far"""
        raise NotImplementedError

    def best_scores(self, how_many=1):
        """Returns `how_many` best scores experienced so far"""
        if how_many < 1:
            how_many = self.max_len
        how_many = min(how_many, self.max_len)
        return self.perf_scores_history[:how_many]


class SparsityAccuracyTracker(TrainingPerformanceTracker):
    """A performance tracker which prioritizes non-zero parameters.

    Sort the performance history using the count of non-zero parameters
    as main sort key, then sort by top1, top5 and and finally epoch number.

    Expects 'top1' and 'top5' to appear in the kwargs.
    """
    def step(self, model, epoch, **kwargs):
        assert all(score in kwargs.keys() for score in ('top1', 'top5'))
        model_sparsity, _, params_nnz_cnt = utils.model_params_stats(model)
        self.perf_scores_history.append(utils.MutableNamedTuple({
            'params_nnz_cnt': -params_nnz_cnt,
            'sparsity': model_sparsity,
            'top1': kwargs['top1'],
            'top5': kwargs['top5'],
            'epoch': epoch}))
        # Keep perf_scores_history sorted from best to worst
        self.perf_scores_history.sort(
            key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'),
            reverse=True)
