import os
import torch
import numpy as np
import glob
import PIL.Image as Image

from torch.nn.functional import one_hot

class LIDC_crops(torch.utils.data.Dataset):
    def __init__(self, img_transform, label_transform, mode='train', data_path='LIDC_crops/LIDC_DLCV_version/'):
        'Initialization'
        self.img_transform = img_transform
        self.label_transform = label_transform
        data_path = os.path.join(data_path, mode)
        self.image_paths = glob.glob(data_path + '/images/*.png')
        self.gt_paths = glob.glob(data_path + '/images/*.png')


        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        split_path = image_path.split('/')
        split_path[3] = 'lesions'
        label_path = os.path.join(*split_path)[:-4] + "_l0.png"

        image = Image.open(image_path)
        label = Image.open(label_path) 
        label = np.array(label)/255

        X = self.img_transform(image)
        y = self.label_transform(label)#.to(torch.long).squeeze(0)

        # y = torch.squeeze(one_hot(y_temp.to(torch.long), 2)).permute(2,0,1)

        return X, y