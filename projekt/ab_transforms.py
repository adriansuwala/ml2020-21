import torch
from torchvision.transforms import functional as F
from torchvision import transforms


class RandomJitter:
    """Resizes input tensor and imgB to scale and then crops to targetscale randomly."""
    def __init__(self, scale, targetscale):
        self.scale = scale
        self.targetscale = targetscale
    
    def __call__(self, imgA, imgB):
        i, j, h, w = transforms.RandomCrop.get_params(imgA, output_size=self.targetscale)
        return F.crop(F.resize(imgA, self.scale), i, j, h, w), F.crop(F.resize(imgB, self.scale), i, j, h, w)


class Resize:
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, imgA, imgB):
        return F.resize(imgA, self.scale), F.resize(imgB, self.scale)


class ToTensor:
    def __call__(self, imgA, imgB):
        return F.to_tensor(imgA), F.to_tensor(imgB)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgA, imgB):
        mean = self.mean
        std = self.std
        return F.normalize(imgA, mean, std), F.normalize(imgB, mean, std)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, imgA, imgB):
        if torch.rand(1) < self.p:
            return F.hflip(imgA), F.hflip(imgB)
        return imgA, imgB


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, imgA, imgB):
        for t in self.transforms:
            imgA, imgB = t(imgA, imgB)
        return imgA, imgB
