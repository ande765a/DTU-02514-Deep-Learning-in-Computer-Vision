import torch
from skimage import transform

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        new_h, new_w = self.output_size
        
        image = transform.resize(image, (new_h, new_w))
        mask = transform.resize(mask, (new_h, new_w))
        
        return {
            "image": image,
            "mask": mask,
        }