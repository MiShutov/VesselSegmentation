import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def weights_init_normal_and_zero_bias(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02) # reset Conv2d's weight(tensor) with Gaussian Distribution
        if hasattr(model, 'bias') and model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0.0) # reset Conv bias(tensor) with Constant(0)
        elif classname.find('BatchNorm3d') != -1:
            torch.nn.init.normal_(model.weight.data, 1.0, 0.02) # reset BatchNorm weight(tensor) with Gaussian Distribution
            torch.nn.init.constant_(model.bias.data, 0.0) # reset BatchNorm bias(tensor) with Constant(0)


def print_imgs_grid(list_of_data, slice_=None, max_imgs_in_a_row=5,
                    titles=[None,], plot_size=4, fontsize=15, is_2d=False, is_color=False):

    n_imgs_in_row = min(list_of_data[0].shape[0], max_imgs_in_a_row)
    fig, ax = plt.subplots(len(list_of_data), 1, figsize=(plot_size*n_imgs_in_row, plot_size*len(list_of_data)))
    
    if len(titles)<len(list_of_data):
        titles += [None,]*(len(list_of_data)-len(titles))
    
    if len(list_of_data)<2:
            print_row_of_images(ax, list_of_data[0], slice_=slice_, max_imgs=max_imgs_in_a_row,
                            min_max=None, title=titles[0], fontsize=fontsize, is_2d=is_2d, is_color=is_color)        
    else:
        for idx, data in enumerate(list_of_data):
            print_row_of_images(ax[idx], data, slice_=slice_, max_imgs=max_imgs_in_a_row,
                            min_max=None, title=titles[idx], fontsize=fontsize, is_2d=is_2d, is_color=is_color)    
    plt.show()


def tensor2img(tensor):
    if len(tensor.shape) == 3:
        img = tensor.permute(1,2,0)
        return img
    else:
        raise RuntimeError("tensor2img: not a 3-dimention")


def print_row_of_images(ax, images, slice_=None, max_imgs=5,
                        min_max=None, title=None, fontsize=None, is_2d=False, is_color=False):
    if slice_==None:
        slice_=int(images[0].shape[-1]/2)
        
    n_imgs_in_row = min(images.shape[0], max_imgs)
    
    if (is_color and is_2d):
        cut_images = images[:n_imgs_in_row, :, : ,:]
        n=cut_images.shape[0]
        c=cut_images.shape[1] 
        h=cut_images.shape[2]
        w=cut_images.shape[3]
        
        frame_size = 2
        delta = w+frame_size
        row_img = torch.zeros((c, h+2*frame_size,
                               n*w + (1+n)*frame_size))
        
        for i in range(n_imgs_in_row):
            row_img[:, frame_size:h+frame_size, frame_size+i*delta:w+frame_size+i*delta] = cut_images[i]
        row_img = tensor2img(row_img)
            
        ax.set_title(title, fontsize=fontsize)
        ax.axis('off')
        im = ax.imshow(row_img)#cmap='seismic')
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=fontsize)
        
    else:
        if is_2d: 
            cut_images = images[:n_imgs_in_row, :, : ,:].squeeze(1)
        else:
            cut_images = images[:n_imgs_in_row, :, : ,:, slice_].squeeze(1)

        n=cut_images.shape[0]
        h=cut_images.shape[1]
        w=cut_images.shape[2]

        frame_size = 2
        row_img = torch.zeros((h+2*frame_size,
                               n*w + (1+n)*frame_size))

        for i in range(n_imgs_in_row):
            delta = w+frame_size
            row_img[frame_size:h+frame_size, frame_size+i*delta:w+frame_size+i*delta] = cut_images[i]

        ax.set_title(title, fontsize=fontsize)
        ax.axis('off')
        im = ax.imshow(row_img, cmap='cool')#cmap="gist_rainbow")#cmap='seismic')
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=fontsize)
    

def get_total_params(model):
    total_params = sum(
    param.numel() for param in model.parameters()
    )
    return(total_params)


def check_None(tensor):
    if torch.isnan(tensor).sum() > 0:
        raise RuntimeError(f"None here ({torch.isnan(tensor).sum()})")


def save_model(model, path_to_save_model, name):
    os.makedirs(path_to_save_model, exist_ok=True)
    torch.save(model.state_dict(), path_to_save_model + "/" + name) 


def load_pretrainned(model, path_to_weights):
    model.load_state_dict(torch.load(path_to_weights))

    