import os
import torch
import numpy as np
import glob
import PIL.Image as Image

from torch.nn.functional import one_hot


def collate_fn(batch):
    print(batch)
    exit()
    return None, None

    #return torch.stack(images), torch.stack(masks)

class LIDC_crops(torch.utils.data.Dataset):
    def __init__(self, base_transform, image_transform, mode='train', data_path='LIDC_crops/LIDC_DLCV_version/', label_version=[0, 1, 2, 3]):
        'Initialization'
        self.base_transform = base_transform
        self.label_version = label_version
        self.image_transform = image_transform
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
        image = Image.open(image_path)

        if isinstance(self.label_version, int):
            label_path = os.path.join(*split_path)[:-4] + "_l" + str(self.label_version) + ".png"
            label = Image.open(label_path)
            X, Y = self.base_transform([image, label])
            X = self.image_transform(X)
            return X, Y
        
        label_paths = [
            os.path.join(*split_path)[:-4] + "_l" + str(version) + ".png"
            for version in self.label_version
        ]
        labels = [Image.open(path) for path in label_paths]
        X, *YS = self.base_transform([image, *labels])
        X = self.image_transform(X)

        return X, torch.stack(YS)