import os
import torch
import numpy as np
import glob
import PIL.Image as Image
from torchvision import datasets
import torchvision.transforms as transforms 

def get_svhn(train, transform, batch_size=64):
    """Get SVHN dataset loader."""
    # image pre-processing
    path = 'SVHN/test/'
    if train:
        path = 'SVHN/train/'
    # dataset and data loader
    svhn_dataset = datasets.SVHN(root=path,
                                 split='train' if train else 'test',
                                 transform=transform,
                                 download=True)
    
    return svhn_dataset
