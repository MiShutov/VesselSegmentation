import numpy as np

def euclidian_map(vol, radius=1):
    old_sizes = np.array(vol.shape)
    new_sizes = old_sizes + 2 * radius
    new_vol = np.zeros(new_sizes)
    
    ### get shifts
    shifts = []
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            for z in range(-radius, radius+1):
                shifts.append(np.array([x, y, z]))

    ### get weigthted sum
    for shift in shifts:
        if (shift[0]==0 and shift[1]==0 and shift[2]==0):
            weight = 2
        else:
            weight = (shift**2).sum()**-0.5
        #print(weight)
        
        coords = [radius+shift[0], radius+shift[1], radius+shift[2]]
        new_vol[coords[0]:old_sizes[0]+coords[0], 
                coords[1]:old_sizes[1]+coords[1],
                coords[2]:old_sizes[2]+coords[2]] += weight*vol
        
    ### apply borders
    new_vol = new_vol[radius:old_sizes[0]+radius,
                      radius:old_sizes[1]+radius,
                      radius:old_sizes[2]+radius]
    new_vol[vol==0]=0
    return(new_vol)


def EDM_norm(edm_vol):
    edm_vol = (edm_vol-edm_vol.min())/(edm_vol.max()-edm_vol.min())
    edm_vol = edm_vol**0.25
    return(edm_vol)