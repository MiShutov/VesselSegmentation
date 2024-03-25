from typing import Dict
from tqdm import tqdm
from utils import weights_init_normal_and_zero_bias, print_imgs_grid
from IPython.display import clear_output

import torch
import torchio as tio

class VG_Controller:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        self.model = config["model"]
        
        otimizers_settings = config["otimizers_settings"]
        self.gen_IS_opt = otimizers_settings['gen_IS_opt'](self.model.gen_IS)
        self.gen_SI_opt = otimizers_settings['gen_SI_opt'](self.model.gen_SI)
        self.disc_I_opt = otimizers_settings['disc_I_opt'](self.model.disc_I)
        self.disc_S_opt = otimizers_settings['disc_S_opt'](self.model.disc_S)
        
        if config.get('sheduler_fn') is not None:
            self.with_sheduler = True
            self.gen_IS_sheduler = otimizers_settings['sheduler_fn'](self.gen_IS_opt)
            self.gen_SI_sheduler = otimizers_settings['sheduler_fn'](self.gen_SI_opt)
            self.disc_I_sheduler = otimizers_settings['sheduler_fn'](self.disc_I_opt)
            self.disc_S_sheduler = otimizers_settings['sheduler_fn'](self.disc_S_opt)
        else:
            self.with_sheduler = False
        
        losses = config["losses"]
        self.I_cycle_loss_fn = losses["I_cycle_loss_fn"]
        self.S_cycle_loss_fn = losses["S_cycle_loss_fn"]
        self.reconstruction_loss_fn = losses["reconstruction_loss_fn"]
        self.segmentation_loss_fn = losses["segmentation_loss_fn"]
        self.discriminator_loss_fn = losses["discriminator_loss_fn"]
        self.generator_loss_fn = losses["generator_loss_fn"]
        self.cycle_lambda = losses["cycle_lambda"]
        self.identity_lambda = losses["identity_lambda"]
        self.reg_lambda = losses["reg_lambda"]
        
        self.epoch = 0
        self.history = None
        self.print_imgs_grid = print_imgs_grid
        
        self.metrics = config["metrics"]
        self.with_supervised = config["with_supervised"]

    def set_initial_weights(self):
        self.model.apply(weights_init_normal_and_zero_bias)
        
        
    def fit(self, dataset, n_epochs):
        self.model = self.model.to(self.device)
        if self.history is None:
            self.history = {
                'train': [],
                'val': [],
                "test": [],
            }
        
        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch+n_epochs):
            self.epoch += 1
            print(f"Epoch {epoch + 1}/{start_epoch+n_epochs}")
            
            train_info = self.train_epoch(dataset.train_dataloader)
            print(train_info)
            self.history['train'].append(train_info)
            
            #if dataset.test_dataloader is not None:
            #    test_info = self.test_epoch(dataset.test_dataloader)
            #    print(test_info)
            #    self.history['test'].append(test_info)
            
            if self.with_sheduler:
                self.gen_IS_sheduler.step()
                self.gen_SI_sheduler.step()
                self.disc_I_sheduler.step()
                self.disc_S_sheduler.step()
            
        return self.model.eval()

    
    def train_epoch(self, train_dataloader):
        
        gen_IS_losses = []
        gen_SI_losses = []
        disc_I_losses = []
        disc_S_losses = []
        segmentation_losses = []
        reconstruction_losses = []
        reg_losses = []
        
        for patches_batch in tqdm(train_dataloader):
            real_I = patches_batch['head']['data'].float().to(self.device)  
            real_S = patches_batch['vessels']['data'].float().to(self.device) 
            
            
            # ----------------
            # Train Generators
            # ----------------
            self.model.gen_IS.train()
            self.model.gen_SI.train()
            self.model.disc_S.eval()
            self.model.disc_I.eval()
            ###GAN Loss
            fake_S = self.model.gen_IS(real_I)
            fake_I = self.model.gen_SI(real_S)
            
            disc_fake_S = self.model.disc_S(fake_S)
            disc_fake_I = self.model.disc_I(fake_I)
            
            gen_IS_loss = self.generator_loss_fn(disc_fake_S)
            gen_SI_loss = self.generator_loss_fn(disc_fake_I)
            
            loss_GAN = gen_IS_loss + gen_SI_loss
            
            ###Cycle Loss
            cycled_S = self.model.gen_IS(fake_I)
            cycled_I = self.model.gen_SI(fake_S)
            
            cycle_loss_I = self.S_cycle_loss_fn(real_S, cycled_S)
            cycle_loss_S = self.I_cycle_loss_fn(real_I, cycled_I)
            loss_cycle = cycle_loss_I + cycle_loss_S
            
            ###Identity Loss
            segmentation_loss = self.segmentation_loss_fn(real_S, cycled_S)
            reconstruction_loss = self.reconstruction_loss_fn(real_I, cycled_I)
            
            loss_identity = segmentation_loss + reconstruction_loss
            
            if self.with_supervised:
                supervised_loss = self.segmentation_loss_fn(real_S, fake_S)
            else:
                supervised_loss = 0
            
            ###
            #reg_loss = torch.pow(fake_S * (1-fake_S), 0.5).mean()
            reg_loss = (torch.log(fake_S+1) * torch.log(2-fake_S)).mean()
            #print(reg_loss)
            ###
            GEN_Loss = loss_GAN + self.cycle_lambda * loss_cycle +\
                       self.identity_lambda * loss_identity + supervised_loss +\
                       self.reg_lambda * reg_loss
            
            self.gen_IS_opt.zero_grad()
            self.gen_SI_opt.zero_grad()
            
            GEN_Loss.backward()
            
            self.gen_IS_opt.step()
            self.gen_SI_opt.step()
            
            
            # --------------------
            # Train Discriminators
            # --------------------
            self.model.gen_IS.eval()
            self.model.gen_SI.eval()
            self.model.disc_S.train()
            self.model.disc_I.train()
            
            fake_S = self.model.gen_IS(real_I)
            fake_I = self.model.gen_SI(real_S)
            
            disc_fake_S = self.model.disc_S(fake_S)
            disc_fake_I = self.model.disc_I(fake_I)
            disc_real_S = self.model.disc_S(real_S)
            disc_real_I = self.model.disc_I(real_I)
            
            disc_I_loss = self.discriminator_loss_fn(disc_real_I, disc_fake_I)
            disc_S_loss = self.discriminator_loss_fn(disc_real_S, disc_fake_S)
            
            self.disc_I_opt.zero_grad()
            self.disc_S_opt.zero_grad()
            
            disc_I_loss.backward()
            disc_S_loss.backward()
            
            self.disc_I_opt.step()
            self.disc_S_opt.step()
            
            ### Add losses to history
            gen_IS_losses.append(gen_IS_loss.item())
            gen_SI_losses.append(gen_SI_loss.item())
            disc_I_losses.append(disc_I_loss.item())
            disc_S_losses.append(disc_S_loss.item())
            segmentation_losses.append(segmentation_loss.item())
            reconstruction_losses.append(reconstruction_loss.item())
            reg_losses.append(reg_loss.item())
            self.model.disc_S.eval()
            self.model.disc_I.eval()
    
        clear_output()
        self.print_imgs_grid([real_I.detach().cpu(), fake_S.detach().cpu(),
                              real_S.detach().cpu(), fake_I.detach().cpu()],
                             titles=["Real Image", "Fake Segmentation", "Real Segmentation", "Fake Image"],
                             plot_size=2, fontsize=15)
        
        self.model.eval()

        out = {'gen_IS_loss': round(sum(gen_IS_losses)/len(gen_IS_losses), 4),
                'gen_SI_loss': round(sum(gen_SI_losses)/len(gen_SI_losses), 4),
                'disc_I_loss': round(sum(disc_I_losses)/len(disc_I_losses), 4),
                'disc_S_loss': round(sum(disc_S_losses)/len(disc_S_losses), 4),
                'segmentation_loss': round(sum(segmentation_losses)/len(segmentation_losses), 4),
                'reconstruction_loss': round(sum(reconstruction_losses)/len(reconstruction_losses), 4),
                'reg_loss': round(sum(reg_losses)/len(reg_losses), 4),
               }
        return out
    
    
    def test_epoch(self, test_dataloader):
        self.model.eval()
        metrics = {"sample" : sample_name}
        for batch in tqdm(test_dataloader):
            patch_loader = batch["patch_loader"]
            grid_aggregator = batch["grid_aggregator"]
            GT = batch["GT"]
            sample_name = batch["sample_name"]
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            for metric_name in self.metrics.keys():
                metrics.update({metric_name : self.metrics[metric_name](GT.data, head_seg)})

        return metrics
    
    
    def fast_predict(self, patch_loader, grid_aggregator, thresh=0.5):
        for patches_batch in patch_loader:
            patch_locations = patches_batch[tio.LOCATION]
            head_patches = patches_batch['head']['data'].to(self.device)
            with torch.no_grad():
                patch_seg = self.model.gen_IS(head_patches)
                grid_aggregator.add_batch(patch_seg.cpu(), patch_locations)
        seg = grid_aggregator.get_output_tensor()
        #seg[seg<thresh]=0
        #seg[seg>0]=1
        return(seg)
    
    
    def predict(self, test_dataloader, path_to_save=None):
        self.model.gen_IS = self.model.gen_IS.to(self.device)
        self.model.gen_IS.eval()
        
        metrics = []
        
        for batch in tqdm(test_dataloader):
            patch_loader = batch["patch_loader"]
            grid_aggregator = batch["grid_aggregator"]
            GT = batch["GT"]
            sample_name = batch["sample_name"]
            head_seg = self.fast_predict(patch_loader, grid_aggregator)
            
            metrics.append({"sample" : sample_name})
            for metric_name in self.metrics.keys():
                metrics[-1].update({metric_name : self.metrics[metric_name](GT.data, head_seg)})
                
            if path_to_save is not None:
                path_to_save_seg = path_to_save + '/' + sample_name + '.nii.gz'
                print(head_seg.sum())
                segImage = tio.Image(tensor=head_seg, affine=GT.affine)
                segImage.save(path_to_save_seg)
        return metrics
    
    
    def single_predict(self, subject, settings):
        grid_sampler = tio.GridSampler(subject,
                                       patch_size=settings["patch_shape"],
                                       patch_overlap=settings["overlap_shape"])
        grid_aggregator = tio.data.GridAggregator(sampler=grid_sampler, overlap_mode='hann')
        patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                   batch_size=settings["batch_size"],
                                                   num_workers=settings["num_workers"])
        seg = self.fast_predict(patch_loader, grid_aggregator)
        return(seg)
    
    
    