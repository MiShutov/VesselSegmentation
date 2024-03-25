import torch
import numpy as np
import warnings

from ml.BaseClasses.ClassMetricMonitor import MetricMonitor
from ml.BaseClasses.WarnigsAndExceptions import InferenceAgentWarning, InferenceAgentError


class InferenceAgent:
    def __init__(self, params=None):
        self.params = params
        if params:
            self._read_params()
        else:
            warnings.warn(f"inited without params!", InferenceAgentWarning)
        
    
    def _read_params(self):
        try:
            if self.params.get('with_warnings', True):
                warnings.simplefilter('default', InferenceAgentWarning) 
            else:
                warnings.simplefilter('ignore', InferenceAgentWarning)
            
            self.device = self.params['device']
            self.metric_functions = self.params.get('metric_functions', {})
            self.metric_monitor = MetricMonitor(metric_functions = self.metric_functions)
        
        except Exception:
            raise InferenceAgentError("_read_params() : bad params")


    def load_from_trainer_state(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        params = checkpoint["trainer_params"]
        self.model = params['model'].to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
    
    
    def load_weights(self, path_to_weights):
        self.model.load_state_dict(torch.load(path_to_weights))
        print(f"Checkpoint loaded from {path_to_weights}")

    
    def predict(self, sample):
        pass

    
    def inference(self, test_loader, thr=None):
        pass