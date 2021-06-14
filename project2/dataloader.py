import os
import torch
import numpy as np
import glob
import PIL.Image as Image
import gdown

class LIDC_crops(torch.utils.data.Dataset):
    def __init__(self, train, transform, mode='train' data_path='LIDC_crops/LIDC_DLCV_version/'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, mode)
        # image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        # image_classes.sort()
        # self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/images/*.png')
        self.gt_paths = glob.glob(data_path + '/images/*.png')

        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        print(image_path[:-4]) 
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y