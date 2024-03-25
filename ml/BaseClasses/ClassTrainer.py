from typing import Dict
from datetime import datetime
import torch
import numpy as np
import os, sys
import copy
import warnings

from ml.metrics import *
from ml.BaseClasses.ClassMetricMonitor import MetricMonitor
from ml.BaseClasses.WarnigsAndExceptions import TrainerWarning, TrainerError


class Trainer:
    """
    trainer_params_example = {
        'device' : "cuda",
        "model": model, #torch.nn.module child
        "loss_fn" : BCELoss(),
        "optimizer" : torch.optim.Adam(model.parameters(), lr=3e-4),
        "scheduler": None, #lambda optimizer: StepLR(optimizer, step_size=5, gamma=0.5)
        "metric_functions" : {
            "F1_fire" : F1_BINARY(),
            "F1_no_fire" : F1_BINARY(inverse=True),
            "PR_fire" : PRECISION_BINARY(),
            "PR_no_fire" : PRECISION_BINARY(inverse=True),
            "RC_fire" : RECALL_BINARY(),
            "RC_no_fire" : RECALL_BINARY(inverse=True),
            "SP_fire" : SPECIFICITY_BINARY(),
            "SP_no_fire" : SPECIFICITY_BINARY(inverse=True),
        }
        "with_warnings": True, #print or not print
        "log_path": f"/beegfs/home/m.shutov/Fires/log/ConvLSTM_{reg}" #path to save
    }
    """
    def __init__(self, params=None):
        self.params = params
        self.recent_epoch = 0
        self.loss_history = {'train': [], 'val': []}
        self.log_path = None
        if params:
            self._read_params()
        else:
            warnings.warn(f"inited without params!", TrainerWarning)

    def _read_params(self):    
        try:
            if self.params.get('with_warnings', True):
                warnings.simplefilter('default', TrainerWarning) 
            else:
                warnings.simplefilter('ignore', TrainerWarning)
            
            self.device = self.params['device']
            self.model = self.params['model'].to(self.device)
            self.optimizer = self.params['optimizer']
            self.loss_fn = self.params['loss_fn']
            
            self.scheduler = self.params.get('scheduler_fn', None)
            self.metric_functions = self.params.get('metric_functions', {})
            self.metric_monitor = MetricMonitor(metric_functions = self.metric_functions)
            if not self.log_path:
                self.log_path = self.params.get('log_path', None)
            if not self.log_path:
                self.log_path = os.path.abspath('./log_') + datetime.now().strftime("%d_%m_%Y_%H:%M")
                warnings.warn(f"log_path is not given! Set self.log_path={self.log_path}",
                              TrainerWarning)
                
            
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

            print("Trainer.log_path:", self.log_path)
        
        except Exception:
            raise TrainerError("_read_params() : bad params")
            
    
    @staticmethod
    def load_model_weights(model, path_to_trainer_state):
        checkpoint = torch.load(path_to_checkpoint)
        model.load_state_dict(torch.load(checkpoint["model_state_dict"]))
        print(f"Model weights loaded from {path_to_trainer_state}")


    def load_trainer_state(self, path_to_checkpoint):
        """
        load trainer state from torch archive 
        """
        checkpoint = torch.load(path_to_checkpoint)
        self.params = checkpoint["trainer_params"]
        self.recent_epoch = checkpoint['epoch']
        self.loss_history = checkpoint["loss_history"]
        self._read_params()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        

    def save_model_weights(self, path, name):
        torch.save(self.model.state_dict(), f"{path}/{name}")
        print(f'Model weights saved at {path}/{name}')
        

    def save_trainer_state(self, path, name):
        """
        save trainer state as torch archive 
        """
        if self.model is None:
            raise RuntimeError("Need a model")

        if not os.path.exists(path):
            os.makedirs(path)
            
        save_params = copy.deepcopy(self.params)
        checkpoint = {
            "trainer_params": save_params,
            "epoch" : self.recent_epoch,
            "loss_history" : self.loss_history,
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "model_state_dict": self.model.state_dict()
        }
        torch.save(checkpoint, f'{path}/{name}')
        

    def stop_event(self):
        #warnings.warn(f"stop event", TrainerWarning)
        date = datetime.now().strftime("%d_%m_%Y_%H:%M")
        save_path = f'{self.log_path}/STOP_EVENTS'
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        self.save_trainer_state(save_path, f'epoch_{self.recent_epoch}_{date}')

    
    def fit(self, **args):
        try:
            self._fit(**args)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        except:
            print("Unexpected error:", sys.exc_info())
        finally:
            self.stop_event()

    
    def _fit(self, *args):
        raise TrainerError("Need to implement _fit(self, *args) function")
        pass
