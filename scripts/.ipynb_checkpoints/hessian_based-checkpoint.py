import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import torch
import torch.nn as nn


def frangi_detect(img, sigma, alpha=0.5, betta=0.5, k=1):
    assert len(img.shape)==3
    H = hessian_matrix(img, sigma, use_gaussian_derivatives=False)
    eigvals = hessian_matrix_eigvals(H)
    sorted_eigvals = eigvals[np.argsort(np.abs(eigvals[:, 0, 0, 0]), 0)]
      
    l_1 = sorted_eigvals[0]
    l_2 = sorted_eigvals[1]
    l_3 = sorted_eigvals[2]

    Sa_sq = (l_2/l_3)**2 #l2^2/l3^2
    Sb_sq = np.abs(l_1)/((np.abs(l_2*l_3))**0.5)    
    S3_sq = l_1**2 + l_2**2 + l_2**2 # l1^2 + l2^2 + l3^2
    
    f1 = 1 - np.exp(-Sa_sq / (2*alpha**2))
    f2 = np.exp(-Sb_sq / (2*betta**2))
    f3 = 1 - np.exp(-S3_sq / (2*k**2))
    
    out = np.zeros_like(l_3)
    out = np.where((l_2 > 0)*(l_3 > 0), 0, f1*f2*f3)
    
    return out  


def hessian_detect_2016(img, sigma, tau=0.5):
    assert len(img.shape)==3
    H = hessian_matrix(img, sigma, use_gaussian_derivatives=False)
    eigvals = hessian_matrix_eigvals(H)
    sorted_eigvals = eigvals[np.argsort(np.abs(eigvals[:, 0, 0, 0]), 0)]
        
    #bright structures on dark background 
    l_1 = -sorted_eigvals[0]
    l_2 = -sorted_eigvals[1]
    l_3 = -sorted_eigvals[2]
    
    M = tau*np.max(l_3) 

    l_rho = np.where((l_3 > 0)*(l_3 <= M), M, l_3)
    l_rho = np.where(l_rho < 0, 0, l_rho)
    
    out = np.zeros_like(l_rho)

    out = np.where((l_2 > 0)*(l_rho > 0), (l_2**2) * (l_rho - l_2) * (27.0/((l_2+l_rho)**3)), 0)
    out = np.where((l_2 > 0)*(l_rho > 0)*(l_2 > l_rho/2), 1, out)
    
    return(out)    



def nn_detect(vol, scale, l_func):
    H = hessian_matrix(vol, scale, use_gaussian_derivatives=False)
    
    eigvals = torch.tensor(np.array(H))
    eigvals = eigvals.permute(1,2,3,0)
    eigvals = nn.Flatten(start_dim=0, end_dim=2)(eigvals)
    eigvals = l_func(eigvals).detach().cpu()
    eigvals = torch.nn.Unflatten(0, vol.shape)(eigvals)
    return eigvals[:, :, :, 0].detach().numpy()