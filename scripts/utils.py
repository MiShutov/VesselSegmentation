import numpy as np
import sys
import os
from time import time, sleep
import matplotlib.pyplot as plt
import re


def get_path(path, key="head"):
    out = []
    names = os.listdir(path)
    for name in names:
        m = re.search(key, name)
        if m:
            out.append(f"{path}/{name}")
    
    if len(out)==1:
        return(out[0])
    return(out)


def print_img(vol, axis, slice_=None, title= 'title', cmap='hot', bar=True, ticks=True, minmax=(None, None)):
    axis.set_title(title)
    im = axis.imshow(vol[:, :, slice_], cmap=cmap, vmin=minmax[0], vmax=minmax[1])
    if bar:
        plt.colorbar(im)
    if not ticks:
        axis.axis('off')


def print_imgs(list_imgs, slice_n=None, size=(5, 4)):
    N = len(list_imgs)
    fig, ax = plt.subplots(1, N, figsize=(N*size[0], size[1]))
    if N==1:
        img = list_imgs[0]
        if len(img.shape)==3:
            img = img[:, :, img.shape[2]//2]
        im = ax.imshow(img)
        plt.colorbar(im)
    else:
        for i,img in enumerate(list_imgs):
            if len(img.shape)==3:
                img = img[:, :, img.shape[2]//2]
            im = ax[i].imshow(img)
            plt.colorbar(im)