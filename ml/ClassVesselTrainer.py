from typing import Dict
import os
import copy
import torch
import torchio as tio
import neptune
from tqdm import tqdm_notebook as tqdm
import torch
from ml.BaseClasses.ClassTrainer import Trainer


class VesselTrainer(Trainer):
    def __init__(self, params=None):
        super(VesselTrainer, self).__init__(params=params)
        self.threshold = 0.5
        self.is2d = params.get('is2d', False)
        self.stop_test_count = params.get('early_stopping', None)
        self.brain_extractor = params.get('brain_extractor', False)
        self.neptune_logger = None


    def log_train_val_epoch(self, save_path):
        pass

    
    def log_saved_epoch(self, path):
        pass

    
    def _fit(self, n_epochs,
             train_loader,
             val_loader=None,
             test_loader=None):
        
        for _ in range(n_epochs):
            self.recent_epoch+=1
            self._train_epoch(train_loader)
            self.log_train_val_epoch(save_path='train')
            self.loss_history['train'].append(
                self.metric_monitor.get_metric_value('Loss', 'avg')
            )
            
            if val_loader:
                with torch.no_grad():
                    self._val_epoch(val_loader)

                self.log_train_val_epoch(save_path='validation')
                self.loss_history['val'].append(self.metric_monitor.get_metric_value('Loss', 'avg'))

            if test_loader:
                with torch.no_grad():
                    self._test_epoch(test_loader)

                self.log_train_val_epoch(save_path='validation')
                self.loss_history['val'].append(self.metric_monitor.get_metric_value('Loss', 'avg'))
                
            ### save  weights ###
            path_to_save = f'{self.log_path}/state_dicts'
            name_to_save = f'state_dict_epoch_{self.recent_epoch}'
            self.save_trainer_state(path_to_save, name_to_save)
            self.log_saved_epoch(f'{path_to_save}/{name_to_save}')
        
            
        print('Finished Training and Validating')
        self.stop_event()

    
    def _train_epoch(self, train_loader):
        self.model.train()
        self.metric_monitor.reset()
        stream = tqdm(train_loader)
        for patches_batch in stream:
            head_batch = patches_batch['head']['data'].float().to(self.device)  
            if self.brain_extractor:
                vessels_batch = patches_batch['brain']['data'].float().to(self.device) 
            else:
                vessels_batch = patches_batch['vessels']['data'].float().to(self.device) 
            
            if self.is2d:
                head_batch = head_batch[:, :, :, :, 0]
                vessels_batch = vessels_batch[:, :, :, :, 0]
            
            output = self.model.forward(head_batch)   
            #output = self.model.forward(head_batch)[0]   
            loss = self.loss_fn(vessels_batch, output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()

            # metrics 
            self.metric_monitor.update("Loss", loss.item())
            self.metric_monitor.compute_metrics(
                output, head_batch, threshold=self.threshold
            )
            #print(self.metric_monitor)
            stream.set_description(
                f"Epoch: {self.recent_epoch}. Train. {self.metric_monitor}"
            )
    
    
    def _val_epoch(self, val_dataloader):
        self.model.eval()
        self.metric_monitor.reset()
        stream = tqdm(val_dataloader)
        # gather outputs and targets for global threshold
        global_output = []
        global_target = []
        global_loss = []

        for patches_batch in stream: 
            head_batch = patches_batch['head']['data'].float().to(self.device)  
            vessels_batch = patches_batch['vessels']['data'].float().to(self.device) 
            with torch.no_grad():
                outputs = self.model.forward(head_batch)   
                loss = self.loss_fn(vessels_batch, outputs)
                # compute metrics with dynamic threshold
            
            self.metric_monitor.update("Loss", loss.item())
            threshold = self.metric_monitor.find_threshold(
                output, targets, self.metric_monitor.metric_functions['F1_fire']
            )
            stream.set_description(
                f"Epoch: {self.recent_epoch}. Validation. \n{self.metric_monitor}, threshold: {threshold:.{3}f}"
            )
            
            # gather outputs and targets
            global_output.append(output)
            global_target.append(targets)
            global_loss.append(loss.item())

        # resulting metrics and threshold on validation
        global_output = torch.cat(global_output, dim=0)
        global_target = torch.cat(global_target, dim=0)
        self.threshold = self.metric_monitor.find_threshold(
            global_output, global_target, self.metric_monitor.metric_functions['F1_fire']
        )
        # reset metrics with values on full validation
        self.metric_monitor.reset()
        self.metric_monitor.update("Loss", sum(global_loss)/len(global_loss))
        self.metric_monitor.compute_metrics(global_output,
                                            global_target)
        
        print(f"RESULT {self.recent_epoch}: Validation.\n {self.metric_monitor}, \nfinal_thr: {self.threshold:.{3}f}")
    

    def _test_epoch(self, test_dataloader):
        self.model.eval()
        self.metric_monitor.reset()
        stream = tqdm(test_dataloader)

        global_output = []
        global_target = []
        
        for batch in stream:
            patch_loader = batch["patch_loader"]
            grid_aggregator = batch["grid_aggregator"]
            GT = batch["GT"].data
            sample_name = batch["sample_name"]
            
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            
            stream.set_description(
                f"Epoch: {self.recent_epoch}. Test."
            )
            
            # gather outputs and targets
            global_output.append(head_seg)
            global_target.append(GT)
        
        global_output = torch.cat(global_output, dim=0)
        global_target = torch.cat(global_target, dim=0)
        self.threshold = self.metric_monitor.find_threshold(
            global_output, global_target, self.metric_monitor.metric_functions['DICE']
        )

        self.metric_monitor.compute_metrics(global_output,
                                            global_target)    
        print(f"RESULT {self.recent_epoch}: Validation.\n {self.metric_monitor}, \nthreshold: {self.threshold:.{3}f}")
    
    
    def fast_predict(self, patch_loader, grid_aggregator, thresh=None):
        for patches_batch in patch_loader:
            patch_locations = patches_batch[tio.LOCATION]
            head_patches = patches_batch['head']['data'].to(self.device)
            if self.is2d:
                head_patches = head_patches[:, :, :, :, 0]
            with torch.no_grad():
                patch_seg = self.model(head_patches)
                if self.is2d:
                    patch_seg = patch_seg.unsqueeze(-1)
                grid_aggregator.add_batch(patch_seg.detach().cpu(), patch_locations)
        
        seg = grid_aggregator.get_output_tensor()
        if thresh is not None: 
            seg = torch.where(seg>thresh, 1, 0)
        return(seg)
          

    def stop_event(self):
        super(VesselTrainer, self).stop_event()
        if self.neptune_loger:
            self.neptune_loger.stop()


    def save_trainer_state(self, path, name):
        self.params['threshold'] = self.threshold
        super(VesselTrainer, self).save_trainer_state(path, name)