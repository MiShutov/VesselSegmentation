import torch
from collections import defaultdict


# class for monitoring metrics while training
class MetricMonitor:
    def __init__(self, round_float=3, metric_functions=None):
        self.metric_functions = metric_functions
        self.round_float = round_float
        self.reset()

    
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    
    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    
    def compute_metrics(self, output, targets, threshold=0.5):
        output = torch.where(output>threshold, 1, 0)
        for m_fn in self.metric_functions:
            self.update(m_fn, self.metric_functions[m_fn](output, targets))


    def get_metric_value(self, metric_name, value_name='avg'):
        if not(value_name in ['val', 'count', 'avg']):
            return None

        metric = self.metrics[metric_name]
        return metric[value_name]

    
    def __str__(self):
        return " | ".join(
          [
              #f'{metric_name}: {m["avg"].item():.{self.round_float}f}'
              f'{metric_name}: {m["avg"]:.{self.round_float}f}'
              for (metric_name, m) in self.metrics.items()
          ]
        )
    
    
    @staticmethod
    def find_threshold(output, targets, metric_fn):
        metric_val_best = 0.0
        thr_best = 0.05
        thds = torch.arange(0.05, 1.0, 0.05)
        for thd in thds:
            metric_val = metric_fn(torch.where(output>thd, 1, 0), targets)
            if metric_val > metric_val_best:
                metric_val_best = metric_val
                thr_best = thd
        
        return thr_best

