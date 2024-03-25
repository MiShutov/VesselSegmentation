import numpy as np
import scripts.integral_volume as iv
from scripts.algorithm_class import Algorithm

class Niblack3d(Algorithm):
    def __init__(self, vol=None, window_size=None,
                 coef_mu=None,
                 coef_sig=None,
                 coef_a=None,
                 thresh=None,
                 weights_out=False,
                 min_ratio=0.9):
        
        super().__init__(vol)
        assert(window_size is not None)
        assert(coef_mu is not None)
        assert(coef_sig is not None)
        assert(coef_a is not None)
        
        self.window_size = window_size
        self.coef_mu = coef_mu
        self.coef_sig = coef_sig
        self.coef_a = coef_a
        self.integr = iv.IntegralVol(vol)
        self.integr_sq = iv.IntegralVol(vol**2)
        self.thresh = thresh
        self.weights_out = weights_out
        if weights_out:
            self.min_ratio = min_ratio
            self.C = (np.e-1)/(1-self.min_ratio)
        
    
    def binarize(self, edges=None, return_sigma=False):
        if edges is None:
            edges = [[0, self.vol.shape[0]],
                     [0, self.vol.shape[1]],
                     [0, self.vol.shape[1]]]
        
        bin_vol = np.zeros((edges[0][1]-edges[0][0],
                            edges[1][1]-edges[1][0],
                            edges[2][1]-edges[2][0])) #, dtype=bool)

        sigmas = []

        for i in range(edges[0][0], edges[0][1]):
            for j in range(edges[1][0], edges[1][1]):
                for k in range(edges[2][0], edges[2][1]):

                    w_s = (self.window_size[0]//2,
                           self.window_size[1]//2,
                           self.window_size[2]//2)

                    f = iv.Point3d(i-w_s[0], j-w_s[1], k-w_s[2]).edit_to_vol(self.vol)
                    s = iv.Point3d(i+w_s[0], j+w_s[1], k+w_s[2]).edit_to_vol(self.vol)

                    T, mu, sigma = self.calc_T(w_s, f, s)
                    sigmas.append([mu, sigma])

                    #if (sigma < 120) and (sigma > 80):
                    #    w_s = (w_s[0]*2, w_s[1]*2, w_s[2]*2)
                    #    f = iv.Point3d(i-w_s[0], j-w_s[1], k-w_s[2]).edit_to_vol(bin_vol)
                    #    s = iv.Point3d(i+w_s[0], j+w_s[1], k+w_s[2]).edit_to_vol(bin_vol)
                    #    T, sigma = self.calc_T(integr, integr_sq, w_s, f, s)

                    
                    if self.weights_out is False:
                        if self.vol[i, j, k]>=T:
                            if (self.thresh is not None):
                                if (self.vol[i, j, k] > self.thresh[0] and
                                    self.vol[i, j, k] < self.thresh[1]):
                                    bin_vol[i - edges[0][0],
                                            j - edges[1][0],
                                            k - edges[2][0]] = 1
                            else:    
                                bin_vol[i - edges[0][0],
                                        j - edges[1][0],
                                        k - edges[2][0]] = 1  
                    else:   
                        ratio = self.vol[i, j, k]/T
                        if ratio<self.min_ratio:
                            bin_vol[i - edges[0][0],
                                    j - edges[1][0],
                                    k - edges[2][0]] = 0
                        else:
                            bin_vol[i - edges[0][0],
                                    j - edges[1][0],
                                    k - edges[2][0]] = np.log(np.e + self.C*(ratio - 1))
                        
        if not return_sigma:
            return(bin_vol)
        else:
            return(sigmas)
    
    def calc_mu(self, integr_vol, p1, p2):
        mu = integr_vol.calculate_sum(p1, p2)
        mu *= (1/self.window_size[0]/self.window_size[1]/self.window_size[2])
        return mu
    
    def calc_T(self, w_s, f, s):
        mu = self.calc_mu(self.integr, f, s)
        mu_sq = self.calc_mu(self.integr_sq, f, s)

        sigma = mu_sq - mu**2
        eps = 1e-6
        sigma += eps

        if sigma<0:
            #print("WARNING! sigma<0")
            sigma = 0
        #assert sigma >= 0

        sigma = sigma**0.5
        T = (self.coef_mu * mu) + (self.coef_sig * sigma) + self.coef_a
        return(T, mu, sigma)
    
