from tqdm import tqdm_notebook as tqdm
import torch
import torchio as tio
from ml.BaseClasses.ClassMetricMonitor import MetricMonitor
from ml.BaseClasses.ClassInferenceAgent import InferenceAgent


class VesselInferenceAgent(InferenceAgent):
    def __init__(self, params):
        super(VesselInferenceAgent, self).__init__(params=params)
        self.threshold = 0.5
        self.is2d = params.get('is2d', False)

    
    def load_from_trainer_state(self, path_to_checkpoint):
        checkpoint = torch.load(path_to_checkpoint)
        params = checkpoint["trainer_params"]
        self.model = params['model'].to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.threshold = params['threshold']

    
    def set_model(self, model):
        self.model = model.to(self.device)

    
    def single_predict(self, subject):
        grid_sampler = tio.GridSampler(subject,
                                       patch_size=self.params["patch_shape"],
                                       patch_overlap=self.params["overlap_shape"])
        grid_aggregator = tio.data.GridAggregator(sampler=grid_sampler, overlap_mode='hann')
        patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                   batch_size=self.params["batch_size"],
                                                   num_workers=self.params["num_workers"])
        seg = self.fast_predict(patch_loader, grid_aggregator)
        return(seg)

    
    def fast_predict(self, patch_loader, grid_aggregator):
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
        if self.threshold is not None: 
            seg = torch.where(seg>self.threshold, 1, 0)
        return(seg)

