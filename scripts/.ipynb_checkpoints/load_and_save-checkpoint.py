from pydicom import dcmread
import pylibjpeg
import os
import numpy as np
import torch
import nibabel as nib


def load_nii_vol(path_to_vol, dtype):
    if not os.path.exists(path_to_vol):
        print("ERROR: <", path_to_vol, "> NOT EXISTS")
        return(None, None)
    vol_file = nib.load(path_to_vol)
    vol = np.array(vol_file.dataobj, dtype=dtype)
    return(vol, vol_file.affine)
    
def get_name_from_path(path_to_sample):
    name = ''
    for i in range(len(path_to_sample)-1, -1, -1):
        if path_to_sample[i] not in ('/', "\\"):
            name = path_to_sample[i] + name
        else:
            return name
    raise RuntimeError("Can't get sample name from path")

                                     
def load_sample_data(path_to_sample, dtype):
    head_data = load_nii_vol(path_to_sample + "/head.nii.gz", dtype=dtype)
    vessels_data = load_nii_vol(path_to_sample + "/vessels.nii.gz", dtype=dtype)
    brain_data = load_nii_vol(path_to_sample + "/brain.nii.gz", dtype=dtype)
    data = {
        "head" : head_data[0],
        "vessels" : vessels_data[0],
        "brain" : brain_data[0],
        "affine" : head_data[1],
        "sample_name" : get_name_from_path(path_to_sample)
    }
    return(data)


def get_dcm_info(path_to_dcm):
    slices_names = os.listdir(path=path_to_dcm)
    datasets_dict = {}
    for name in slices_names:
        dcm_file = dcmread(path_to_dcm + name)
        if hasattr(dcm_file, 'SeriesDescription'):
            sd = dcm_file.SeriesDescription
        else:
            sd = "Noname"           
        if (datasets_dict.get(sd) is None):
            datasets_dict.update({
                sd : {
                    "dcm_file": dcm_file,
                    "vol_shape" : None,
                    "vox_size" : None,
                    "file_names" : [name,]
                }    
            })
        else:
            datasets_dict[sd]["file_names"].append(name)
    for key in datasets_dict.keys():
        dcm_file = dcmread(path_to_dcm + datasets_dict[key]["file_names"][0])
        rows = dcm_file.Rows
        columns = dcm_file.Columns
        depth = len(datasets_dict[key]["file_names"])
        sizes = (rows, columns, depth)
        datasets_dict[key]["vol_shape"] = sizes
         
        if hasattr(dcm_file, 'SpacingBetweenSlices') and hasattr(dcm_file, "PixelSpacing"):
            voxel_size = (dcm_file.PixelSpacing[0], dcm_file.PixelSpacing[1], dcm_file.SpacingBetweenSlices)
            datasets_dict[key]["vox_size"] = voxel_size
            
    return(datasets_dict)


def get_dcm_vol(path_to_slices, study_dict):
    vol_shape = study_dict["vol_shape"] 
    vox_size = study_dict["vox_size"]        
    
    dcm_file = dcmread(path_to_slices + study_dict["file_names"][0])
    dtype = type(dcm_file.pixel_array[0][0])
    vol = np.zeros(vol_shape, dtype=dtype)
    
    for indx, name in enumerate(study_dict["file_names"]):
        #print(indx, name)
        dcm_file = dcmread(path_to_slices + name)
        
        vol[:, :, indx] = dcm_file.pixel_array
    
    return(vol, vox_size)


def vox_size2affine(vox_size):
    return [[vox_size[0], 0., 0., 0.],
            [0., vox_size[1], 0., 0.],
            [0., 0., vox_size[2], 0.],
            [0., 0., 0., 1.]]


def save_vol_as_nii(arr, affine, path_to_save):
    if len(arr.shape) not in (3, 4):
        raise "Error::save_vol_as_nii: bad array shape"
    if len(arr.shape)==4:
        arr = arr[0]
    if type(arr) is torch.Tensor:
        arr = arr.numpy()
    empty_header = nib.Nifti1Header()
    Nifti1Image = nib.Nifti1Image(arr, affine, empty_header)
    nib.save(Nifti1Image, path_to_save)

    
def raw2nifti(path_to_row, path_to_nifti, shape, dtype, return_vol=False):
    arr = np.fromfile(path_to_row, dtype=dtype).reshape(shape)
    affine = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],]
    save_vol_as_nii(arr, affine, path_to_nifti)
    if return_vol:
        return arr


