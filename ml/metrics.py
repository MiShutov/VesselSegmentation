import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.pytorch_ssim import SSIM3D
from ml.soft_skeleton import soft_skel


class TverskyLoss:
    def __init__(self, beta, discrepancy=0.5):
        self.discrepancy = discrepancy
        self.beta = beta

    def __call__(self, y_real, y_pred):   
        num = torch.sum(y_real*y_pred) + self.discrepancy
        den = num + self.beta * torch.sum( (1 - y_real) * y_pred) + \
              (1 - self.beta) * torch.sum(y_real * (1 - y_pred))
        res = 1 - (num/den)
        return res 
    

class DiceLoss:
    def __init__(self, discrepancy=1):
        self.discrepancy = discrepancy

    def __call__(self, y_real, y_pred):
        num = 2*torch.sum(y_real*y_pred) + self.discrepancy
        den = torch.sum(y_real + y_pred) + self.discrepancy
        res = 1 - (num/den)
        return res 


class WeightedExpBCE:
    def __init__(self, gamma_bce, bce_weight=1, eps=1e-8):
        self.bce_weight = bce_weight
        self.gamma_bce = gamma_bce
        self.eps = 1e-8

    def set_bce_weight(self, freq):
        assert freq > 0
        assert freq < 1
        w1 = (1 / freq) ** 0.5
        w2 = (1 / (1 - freq) ) ** 0.5
        self.bce_weight = w1 / w2
    
    def __call__(self, y_real, y_pred):
        first_term = torch.pow(- y_real * torch.log(y_pred + self.eps) + self.eps, self.gamma_bce)
        second_term = - (1 - y_real) * torch.log(1 - y_pred + self.eps)
        second_term = torch.pow(second_term + self.eps, self.gamma_bce)
        return torch.mean(self.bce_weight * first_term + second_term)
    

class ExponentialLogarithmicLoss:
    def __init__(self, gamma_tversky = None, gamma_bce = None,
                 freq=None, lamb=0.5, tversky_alfa=0.5):
        assert gamma_tversky is not None
        assert freq is not None
        if gamma_bce is None:
            gamma_bce = gamma_tversky
        self.lamb = lamb
        self.weighted_bce_loss = WeightedExpBCE(gamma_bce)
        self.weighted_bce_loss.set_bce_weight(freq)
        self.gamma_tversky = gamma_tversky
        self.tversky = TverskyLoss(tversky_alfa)
        self.eps = 1e-8
        
        
    def __call__(self, y_real, y_pred):
        w_exp_bce = self.weighted_bce_loss(y_real, y_pred)
        log_tversky = -torch.log(1-self.tversky(y_real, y_pred) + self.eps)
        epx_log_tversky = torch.pow(log_tversky + self.eps, self.gamma_tversky)
        #print("w_exp_bce:", w_exp_bce, "epx_log_tversky:", epx_log_tversky)
        return self.lamb * w_exp_bce + (1 - self.lamb) * epx_log_tversky

    
class SumLoss:
    def __init__(self, alfa=0.5):
        self.eps = 1e-3
        self.alfa = alfa
        
    def __call__(self, y_real, y_pred):
        k = y_real.sum()/y_pred.sum()
        k = torch.clip(k, min=0.1, max=10)
        #print("k:", k)
        loss = torch.log(self.alfa * torch.exp(k) + (1-self.alfa) * torch.exp(1/(k+self.eps)))-1
        return loss
    

class LinearCombLoss:
    def __init__(self, funcs_and_сoef_list):
        self.funcs_and_сoef_list = funcs_and_сoef_list
        
    def __call__(self, y_real, y_pred):
        loss = 0
        for func, coef in self.funcs_and_сoef_list:
            f_i = func(y_real, y_pred)
            #print("f_i:", f_i)
            loss += coef * func(y_real, y_pred)
            
        return loss
    
    
class MultyscaleLoss(nn.Module):
    def __init__(self, loss):
        super(MultyscaleLoss, self).__init__()
        self.loss = loss
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, y_real, outs):
        out_loss = 0
        for idx, out in enumerate(outs):
            #out_loss += 2**(-3 * idx) * self.loss(y_real, out)
            out_loss += 2**(-idx) * self.loss(y_real, out)
            y_real = self.pool(y_real)
            
        return(out_loss)    


def minimax_norm(tensor):
    normed = (tensor - tensor.min())/(tensor.max() - tensor.min() + 1e-8)
    return normed


class CycleLoss:
    def __init__(self, loss_type='bce'):
        if loss_type=='bce':
            self.loss_fn = nn.BCELoss()
        if loss_type=='cldice':
            self.loss_fn = soft_dice_cldice()
        if loss_type=='mse':
            self.loss_fn = nn.MSELoss()
        if loss_type=='ssim':
            self.loss_fn = SSIM3D()
        if loss_type=='mae':
            self.loss_fn = nn.L1Loss()
        if loss_type=='tversky':
            self.loss_fn = TverskyLoss(beta=0.2)
        if loss_type=='explog':
            self.loss_fn = ExponentialLogarithmicLoss(gamma_tversky=1, gamma_bce=1,
                                                      freq=0.1, lamb=0.5, tversky_alfa=0.5)
            
    def __call__(self, real, fake):
        #real = minimax_norm(real)
        #fake = minimax_norm(fake)
        return self.loss_fn(real, fake)


class MSE_normed: #!!!Apply only to minimax-normed images  
    def __init__(self, discrepancy=1e-8):
        self.discrepancy = discrepancy

    def __call__(self, y_real, y_fake):
        x = (y_fake/(y_real+self.discrepancy))
        res = (1-x)*(1-x)
        return res 

    
class DiscriminatorLoss:
    def __init__(self, type=None):
        self.loss_fn = nn.MSELoss()
        self.afpha = 0.5

    def __call__(self, real, fake):
        #print("real:", real )
        #print("fake:", fake )
        #print(self.afpha * self.loss_fn(torch.ones_like(real), real) +\
        #       (1 - self.afpha) * self.loss_fn(torch.zeros_like(fake), fake))
        
        return self.afpha * self.loss_fn(torch.ones_like(real), real) +\
               (1 - self.afpha) * self.loss_fn(torch.zeros_like(fake), fake)


class GeneratorLoss:
    def __init__(self, type=None):
        self.loss_fn = nn.MSELoss()

    def __call__(self, dicriminator_response_to_fake):
        #print("dicriminator_response_to_fake:", dicriminator_response_to_fake.mean() )
        return (self.loss_fn(torch.ones_like(dicriminator_response_to_fake), dicriminator_response_to_fake))  


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth) 
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice
    

class soft_dice_cldice(nn.Module):
    def __init__(self, alpha=0.5, iters=3, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.cldice = soft_cldice(iters, smooth)
        self.dice = DiceLoss(smooth)
        self.alpha = alpha
        
    def forward(self, y_true, y_pred):
        return self.alpha*self.cldice(y_true, y_pred) + (1-self.alpha)*self.cldice(y_true, y_pred)


    
class MIP_tool():
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.projections = {}

    def add_projection(self, vol, name, axis=2):
        if len(vol.shape)!=3:
            if len(vol.shape) == 4:
                vol = vol[0]
            elif len(vol.shape) == 5:
                vol = vol[0, 0]
            else:    
                raise RuntimeError("MIP::add_projection: bad vol!")
        
        proj = torch.max(vol, axis)
        projections.updade({name: proj})
        
        
        
    def get_projection(self, name):
        proj = self.projections.get(name, None)
        return(proj)
    
    
### NON-fuzzy metrics ##
    
class IOU():
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()
        labels = labels.byte()
        
        intersection = (outputs & labels).float().sum()  
        union = (outputs | labels).float().sum()

        iou = (intersection + self.eps) / (union + self.eps)
        return iou   


class JAC_BINARY():
    def __init__(self, eps = 1e-8):
        self.eps = eps

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        
        outputs = outputs.byte()  
        labels = labels.byte()
        
        TP = (outputs & labels).float().sum((1, 2, 3))  
        FN = ((1-outputs) & labels).float().sum()
        FP = (outputs & (1-labels)).float().sum()
        
        jac = (TP + self.eps) / (TP + FN + FP + self.eps)
        return jac
    

class DICE_BINARY():
    def __init__(self, eps=1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()  
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
            
        TP = (outputs & labels).float().sum()  
        FN = ((1-outputs) & labels).float().sum()
        FP = (outputs & (1-labels)).float().sum()
        
        dice = (2*TP + self.eps) / (2*TP + FN + FP + self.eps)
        return dice


class F1_BINARY():
    def __init__(self, eps=1e-8, inverse=False):
        self.dice = DICE_BINARY(inverse=inverse)

    def __call__(self, outputs, labels):
        return self.dice(outputs, labels)


class SENSIVITY_BINARY():
    def __init__(self, eps = 1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()  
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
        
        TP = (outputs & labels).float().sum()  
        FN = ((1-outputs) & labels).float().sum()
        
        sn = (TP + self.eps) / (TP + FN + self.eps)
        return sn
    

class SPECIFICITY_BINARY():
    def __init__(self, eps = 1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse

    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte() 
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
        
        TN = ((1-outputs) & (1-labels)).float().sum()  
        FP = (outputs & (1-labels)).float().sum()
        
        sp = (TN + self.eps) / (TN + FP + self.eps)
        return sp
    

class PRECISION_BINARY():
    def __init__(self, eps = 1e-8, inverse=False):
        self.eps = eps
        self.inverse = inverse
    
    def __call__(self, outputs, labels):
        assert outputs.shape==labels.shape
        outputs = outputs.byte()  
        labels = labels.byte()
        if self.inverse:
            outputs = 1-outputs
            labels = 1-labels
        
        TP = (outputs & labels).float().sum()  
        FP = (outputs & (1-labels)).float().sum()
        
        pr = (TP + self.eps) / (TP + FP + self.eps)
        return pr


class RECALL_BINARY():
    def __init__(self, eps = 1e-8, inverse=False):
        self.eps = eps
        self.sn = SENSIVITY_BINARY(inverse=inverse)

    def __call__(self, outputs, labels):
        return self.sn(outputs, labels)
    
    