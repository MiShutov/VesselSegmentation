import numpy as np
import time
import cv2

class Point3d:
    def __init__(self, x : int, y : int, z : int):
        self.x = x
        self.y = y
        self.z = z
    
    def edit_to_shape(self, shape):
        self.x = min(max(self.x, 0), shape[0]-1)
        self.y = min(max(self.y, 0), shape[1]-1)
        self.z = min(max(self.z, 0), shape[2]-1)
        return(self)
    
    def edit_to_vol(self, vol):
        self.x = min(max(self.x, 0), vol.shape[0]-1)
        self.y = min(max(self.y, 0), vol.shape[1]-1)
        self.z = min(max(self.z, 0), vol.shape[2]-1)
        return(self)
        
    def __str__(self) -> str:
        return f"x: {self.x}, y: {self.y}, z: {self.z}"


class IntegralVol:
    def __init__(self, vol: np.array):
        
        self.integral_vol = np.zeros((vol.shape[0]+1, vol.shape[1]+1, vol.shape[2]+1))
        for i in range(1, self.integral_vol.shape[-1]):
            self.integral_vol[:, :, i] = self.integral_vol[:, :, i-1] + cv2.integral(vol[:,:,i-1])
        #print("3d integral volume calculated")   
           
    
    #point1 : first coord of the volume [x, y, z]
    #point2 : second coord of the volume [x, y, z]
    def calculate_sum(self, point1, point2): 
        f = point1
        s = Point3d(point2.x+1, point2.y+1, point2.z+1)

        vol_sum = self.integral_vol[s.x, s.y, s.z] - \
                  self.integral_vol[s.x, s.y, f.z] - \
                  self.integral_vol[f.x, s.y, s.z] - \
                  self.integral_vol[s.x, f.y, s.z] + \
                  self.integral_vol[s.x, f.y, f.z] + \
                  self.integral_vol[f.x, s.y, f.z] + \
                  self.integral_vol[f.x, f.y, s.z] - \
                  self.integral_vol[f.x, f.y, f.z]
        
        return(vol_sum)
    

def IntegralVolTest():
    test_vol = np.random.rand(500, 500, 200)
    
    start = time.time()
    test_vol_integr = IntegralVol(test_vol)
    end = time.time()
    print("preprocess time", end - start)
    
    point1 = Point3d(0, 0, 0)
    point2 = Point3d(249, 249, 174)


    start = time.time()
    for i in range(1000):
        test_vol[point1.x:point2.x+1, point1.y:point2.y+1, point1.z:point2.z+1].sum()
    end = time.time()
    print("numpy time", end - start)


    start = time.time()
    for i in range(1000):
        test_vol_integr.calculate_sum(point1, point2)
    end = time.time()
    print("integral vol time", end - start)
    
    
    errors = []
    iters = 100
    for i in range(iters):
        point1 = Point3d(np.random.randint(0, 100),
                         np.random.randint(0, 100),
                         np.random.randint(0, 100))
        point2 = Point3d(np.random.randint(125, 249),
                         np.random.randint(125, 249),
                         np.random.randint(100, 150))
        nump = test_vol[point1.x:point2.x+1, point1.y:point2.y+1, point1.z:point2.z+1].sum()
        integr = test_vol_integr.calculate_sum(point1, point2)
        #print(abs(nump-integr))
        errors.append(abs(nump-integr))
    print("calculation error:", max(errors))
    
