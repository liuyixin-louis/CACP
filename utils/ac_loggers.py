
import csv


__all__ = ["CACPStatsLogger", "FineTuneStatsLogger"]


class _CSVLogger(object):
    def __init__(self, fname, headers):
        """Create the CSV file and write the column names"""
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        self.fname = fname

    def add_record(self, fields):
        # We close the file each time to flush on every write, and protect against data-loss on crashes
        with open(self.fname, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)


class CACPStatsLogger(_CSVLogger):
    def __init__(self, fname):
        headers = ['episode', 'top1', 'reward', 'total_macs', 'normalized_macs', 'normalized_nnz',
                   'ckpt_name', 'action_history', 'agent_action_history', 'performance',"pruning_ratio"]
        super().__init__(fname, headers)


class FineTuneStatsLogger(_CSVLogger):
    def __init__(self, fname):
        headers = ['episode', 'ft_top1_list']
        super().__init__(fname, headers)
