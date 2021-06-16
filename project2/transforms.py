import math
import warnings
from typing import List, Sequence, Tuple
from numpy.lib.function_base import interp
import torch
from skimage import transform

from random import random
from torch import tensor
import torchvision 
import torch.functional as F

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, _interpolation_modes_from_int


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


class MultiHorizontalFlip():
    def __init__(self):
        super(MultiHorizontalFlip)

    def __call__(self, images):
        if random() < 0.5:
            return images

        return [TF.hflip(image) for image in images]


class MultiRandomRotation():
    def __init__(self, degrees, **kwargs):
        self.degrees = degrees
        self.kwargs = kwargs
    
    def __call__(self, images):
        angle = (random() * 2 - 1) * self.degrees
        return [TF.rotate(image, angle, **self.kwargs) for image in images]


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(image: tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(image)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, images):
        return [TF.resized_crop(image, self.size, self.interpolation, self.max_size, self.antialias) for image in images]


class MultiToTensor():
    def __call__(self, images):
        return [TF.to_tensor(image) for image in images]


class MultiNormalize():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, images):
        return [TF.normalize(image, **self.kwargs) for image in images]