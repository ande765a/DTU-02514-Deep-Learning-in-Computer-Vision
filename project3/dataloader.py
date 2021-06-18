import os
import torch
import numpy as np
import glob
import PIL.Image as Image


class horse2zebra(torch.utils.data.Dataset):
    def __init__(self, base_transform, train=True, data_path='horse2zebra/'):
        'Initialization'
        mode = 'train'
        if not train:
            mode = 'test'
        self.base_transform = base_transform
        data_path = os.path.join(data_path, mode)
        self.horse_paths = glob.glob(data_path + '/horse/*.png')
        self.zebra_paths = glob.glob(data_path + '/zebra/*.png')

    def __len__(self):
        'Returns the total number of samples'
        return len(self.zebra_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        
        horse_path = self.horse_paths[idx % len(self.horse_paths)]
        zebra_path = self.zebra_paths[idx % len(self.zebra_paths)]
        
        horse = Image.open(horse_path).convert('RGB')
        zebra = Image.open(zebra_path).convert('RGB')
        horse = 2 * (horse / np.max(horse)) - 1
        zebra = 2 * (zebra / np.max(zebra)) - 1
        
        H = self.transform(horse)
        Z = self.transform(zebra)
       
        return H, Z

