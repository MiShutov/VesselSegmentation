import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from scripts.load_and_save import load_sample_data


class RAMdataset(Dataset):
    def __init__(self, settings):
        super(Dataset, self).__init__()
        assert settings["mode"] in ('train', 'eval')
        self.mode = settings["mode"]
        self.data_dir = settings["data_dir"]
        self.patch_data = pd.read_csv(settings["data_dir"] + '/patch_data.csv')
        self.patch_shape = settings['patch_shape']
        self.sample_data = pd.read_csv(settings["data_dir"] + '/sample_data.csv')
        self.RAM_samples = settings["RAM_samples"]
        

    def __len__(self):
        if self.mode=='train':
            return len(self.patch_data)
        if self.mode=='eval':
            return len(self.sample_data)

    def __getitem__(self, idx):
        if self.mode=='train':
            patch_info = self.patch_data.iloc[idx]
            if (self.RAM_samples):
                head_vol = norm_vol(self.RAM_samples[patch_info.sample_name]["head"])
                vessels_vol = self.RAM_samples[patch_info.sample_name]["vessels"]
                brain_vol = self.RAM_samples[patch_info.sample_name]["brain"]
            else: 
                path_to_sample = self.sample_data[
                    self.sample_data.sample_name == patch_info.sample_name] \
                    .iloc[0] \
                    .sample_path
                sample_data = load_sample_data(path_to_sample, np.float32)
                head_vol = norm_vol(sample_data["head"])
                vessels_vol = sample_data["vessels"]
                brain_vol = sample_data["brain"]
            
            head_patch = self.get_patch(patch_info, head_vol)
            vessels_patch = self.get_patch(patch_info, vessels_vol)
            brain_patch = self.get_patch(patch_info, brain_vol)
            return {'head_patch': head_patch, 'vessels_patch': vessels_patch, 'brain_patch': brain_patch}    
            
        if self.mode=='eval':
            sample_info = self.sample_data.iloc[idx]
            if (self.RAM_samples):
                sample_data = self.RAM_samples[sample_info.sample_name]
            else: 
                sample_data = load_sample_data(sample_info.sample_path, np.float32)
            sample_data["head"] = torch.tensor(norm_vol(sample_data["head"])).unsqueeze(0)
            sample_data["vessels"] = torch.tensor(sample_data["vessels"]).unsqueeze(0)
            sample_data["brain"] = torch.tensor(sample_data["brain"]).unsqueeze(0)
            return sample_data
            

    def get_patch(self, patch_info, vol):  
        (x, y, z) = (int(patch_info.pixel_x),
                     int(patch_info.pixel_y),
                     int(patch_info.pixel_z))
        ps = self.patch_shape
        patch = torch.tensor(vol[x:x+ps[0],
                                 y:y+ps[1],
                                 z:z+ps[2]]).unsqueeze(0)
        return(patch)
    
    def np2torch(self, np_arr):
        return(torch.tensor(np_arr).unsqueeze(0).unsqueeze(0))
        
    
def generate_patches_pixels(vol_shape, patch_shape, patches_number):
    np.random.seed(1608)
    patch_pixel_x = []
    patch_pixel_y = []
    patch_pixel_z = []
    for i in range(patches_number):
        x = np.random.randint(low=0, high=vol_shape[0]-patch_shape[0])
        y = np.random.randint(low=0, high=vol_shape[1]-patch_shape[1])
        z = np.random.randint(low=0, high=vol_shape[2]-patch_shape[2])
        patch_pixel_x.append(x)
        patch_pixel_y.append(y)
        patch_pixel_z.append(z)
    return(patch_pixel_x, patch_pixel_y, patch_pixel_z)


def norm_vol(vol, mode="linear"): #mode= "linear", "normal"
    assert mode in ("linear", "normal")
    if mode == "normal":
        vol = (vol-vol.mean())/vol.std() # -1, 1
    if mode == "linear":
        vol = (vol-vol.min())/(vol.max() - vol.min()) #from 0 to 1
    return vol


def preprocess_dataset(settings, dtype=np.float32):
    patch_data_df = {"pixel_x" : [],
                     "pixel_y" : [],
                     "pixel_z" : [],
                     "sample_name" : []}
    patch_data_df = pd.DataFrame(patch_data_df)        
    patch_data_df = patch_data_df.astype({"pixel_x": int, "pixel_y": int, "pixel_z": int})
    
    sample_paths_list = []
    sample_names_list = []
    
    if settings["RAM_samples"] != False:
        settings["RAM_samples"] = {}
    
    for dirname, dirnames, filenames in os.walk(settings['data_dir']):
        for subdirname in dirnames:
            sample_paths_list.append(os.path.join(dirname, subdirname))
            sample_names_list.append(subdirname)
            sample = load_sample_data(sample_paths_list[-1], dtype)
            check_shapes_of_data(sample, sample_paths_list[-1])
            if settings["RAM_samples"] != False:
                settings["RAM_samples"].update({sample_names_list[-1] : sample})
            
            sample_patch_pixels = generate_patches_pixels(sample['head'].shape,
                                                          settings['patch_shape'],
                                                          settings['number_of_patches'])
            sample_names = settings['number_of_patches'] * [sample_names_list[-1],]
            sample_df = pd.DataFrame({"pixel_x" : sample_patch_pixels[0],
                                      "pixel_y" : sample_patch_pixels[1],
                                      "pixel_z" : sample_patch_pixels[2],
                                      "sample_name" : sample_names})
            patch_data_df = pd.concat([patch_data_df, sample_df], ignore_index=True)
    
    patch_data_df.to_csv(settings['data_dir'] + "/patch_data.csv", index=False)    
    sample_data_df = pd.DataFrame({"sample_name" : sample_names_list,
                                    "sample_path" : sample_paths_list})
    sample_data_df.to_csv(settings['data_dir'] + "/sample_data.csv", index=False)
    return(patch_data_df, sample_data_df)


