import os
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm
import re


class DataProcessor:
    def __init__(self, aug_coef=2, resample=None):
        intepolate = "hamming"#'lanczos' # 'bspline'   
        transforms = [
            #tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
            tio.transforms.ZNormalization(),
            ]
        
        aug_transforms = [
            #tio.transforms.RandomBiasField(0.1),
            tio.transforms.RandomGamma((-0.07, 0.07)),
            tio.transforms.RandomElasticDeformation(num_control_points=7,
                                                    max_displacement=7.5,
                                                    image_interpolation=intepolate),
            #tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
            tio.transforms.ZNormalization(),
            ]
        if resample is not None:
            transforms = [tio.transforms.Resample(target=resample,
                                                  image_interpolation=intepolate,
                                                  label_interpolation=intepolate),] + transforms
            aug_transforms = [tio.transforms.Resample(target=resample,
                                                      image_interpolation=intepolate,
                                                      label_interpolation=intepolate),] + aug_transforms
        
        self.transform = tio.Compose(transforms)
        self.aug_transform = tio.Compose(aug_transforms)
        
        self.aug_coef = aug_coef
        self.names_dict = {}
        
    
    def remove_affine_shift(self, affine):
        affine[0][3] = 0
        affine[1][3] = 0
        affine[2][3] = 0
        return(affine)
    
    
    def save_subject(self, subject, save_dir):
        if subject.sample_name not in self.names_dict.keys():
            self.names_dict.update({subject.sample_name : 0})
        else:
            self.names_dict[subject.sample_name]+=1
        path = save_dir + f"/{subject.sample_name}_{self.names_dict[subject.sample_name]}"
        if not os.path.exists(path):
            os.mkdir(path)
        if 'head' in subject.keys():
            self.remove_affine_shift(subject.head.affine)
            subject.head.save(path+"/head.nii.gz")  
        if 'vessels' in subject.keys():
            self.remove_affine_shift(subject.vessels.affine)
            subject.vessels.save(path+"/vessels.nii.gz")
        if 'brain' in subject.keys():
            self.remove_affine_shift(subject.brain.affine)
            subject.brain.save(path+"/brain.nii.gz")
    
    def __call__(self, raw_data_path, processed_data_path):
        def get_path(path, key="head"):
            names = os.listdir(path)
            for name in names:
                m = re.search(key, name)
                if m:
                    return(path + '/' + name)  

        subjects_list = []
        for dirname, dirnames, filenames in os.walk(raw_data_path):
            for subdirname in dirnames:
                print(subdirname)
                p = os.path.join(dirname, subdirname)
                subject_dict = {"sample_name" : subdirname}
                    
                if get_path(p, "head"):
                    subject_dict.update({'head': tio.ScalarImage(get_path(p, "head"))})
                if get_path(p, "vessels"):
                    subject_dict.update({'vessels': tio.LabelMap(get_path(p, "vessels"))})
                if get_path(p, "brain"):
                    subject_dict.update({'brain': tio.LabelMap(get_path(p, "brain"))})
                subject = tio.Subject(subject_dict)
                subjects_list += (subject,)
        #print(subjects_list)
        
        processed_subjects_list = []
        for subject in tqdm(subjects_list):
            for i in range(self.aug_coef):
                if i==0: #without transforms
                    subject = self.transform(subject)
                else:
                    subject = self.aug_transform(subject)
                self.save_subject(subject, processed_data_path)