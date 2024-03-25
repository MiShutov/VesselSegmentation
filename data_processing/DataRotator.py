import os
import numpy as np
import torch
import torchio as tio
import matplotlib.pyplot as plt


class DataRotator:
    def __init__(self, path_to_sample, new_file=True):
        self.path = path_to_sample
        self.new_file = new_file
        self.filenames = []
        for name in os.listdir(path_to_sample):
            if name[-7:]==".nii.gz":
                self.filenames.append(name)
        if (len(self.filenames)==0):
            raise RuntimeError("DataRotator::ERROR: not any vol files")
        self.vol_data_dict = {}
        for name in self.filenames:
            if name=="vessels.nii.gz":
                self.vol_data_dict.update({name : tio.LabelMap(self.path + '/' + name)})
            else:
                self.vol_data_dict.update({name : tio.ScalarImage(self.path + '/' + name)})
        self.orientation = 0
        
    
    def show_vol(self, name, axis, slice_to_show, cmap='hot'):
        axis.set_title(name)
        im = axis.imshow(self.vol_data_dict[name].data[0, :, :, slice_to_show], cmap=cmap)
        plt.colorbar(im)
        
    
    def show(self, slice_to_show):
        N = len(self.filenames)
        fig, ax = plt.subplots(1, N, figsize=(4*N, 4))
        for idx, name in enumerate(self.filenames):
            if N>1:
                self.show_vol(name, ax[idx], slice_to_show)
            elif N==1:
                self.show_vol(name, ax, slice_to_show)
    
    
    def transpose(self):
        for name in self.filenames:
            self.vol_data_dict[name].set_data(torch.transpose(self.vol_data_dict[name].data, 1, 2))
    
    def mirror(self):
        for name in self.filenames:
            self.vol_data_dict[name].set_data(torch.flip(self.vol_data_dict[name].data, [1, 2]))
        
    def save(self):
        if self.new_file:
            if not os.path.exists(self.path + "/rotated"):
                os.mkdir(self.path + "/rotated")
            path_to_save = self.path + "/rotated"
        else:
            path_to_save = self.path
    
        for name in self.filenames:        
            self.vol_data_dict[name].save(path_to_save + '/' + name)
            print(f"Volume <{name}> saved.")