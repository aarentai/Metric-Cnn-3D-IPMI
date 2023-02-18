import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
from skimage import data, filters


torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DatasetHCP(Dataset):
    def __init__(self, data_dir, sample_name_list):   
        self.data_dir = data_dir
        self.sample_name_list = sample_name_list
        self.sample_name_list.sort()
        
        sample_name = self.sample_name_list[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vector_field_path = f'{self.data_dir}/{sample_name}/{sample_name}_shrinktensor_principal_vector_field.nhdr'
        mask_path = f'{self.data_dir}/{sample_name}/{sample_name}_shrinktensor_filt_mask.nhdr'
        self.vector_field = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(vector_field_path))).to(device)*1000.0
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        self.mask = torch.from_numpy((self.mask)).to(device)
        self.mask[0]=0
        self.mask[-1]=0
        self.mask[:,0]=0
        self.mask[:,-1]=0
        self.mask[:,:,0]=0
        self.mask[:,:,-1]=0
        
    def __len__(self):
        return len(self.sample_name_list)
        
    def __getitem__(self, idx):
        sample = {  'vector_field'  : self.vector_field.float(),
                    'mask'          : self.mask.unsqueeze(0).float()}
        return sample
    
    