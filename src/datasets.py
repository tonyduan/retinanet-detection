import torch
import random
from torchvision import datasets, transforms


class PrecisionTransform(object):
    
    def __init__(self, precision):
        self.precision = precision

    def __call__(self, x):
        if self.precision == "float":
            return x.float()
        if self.precision == "half":
            return x.half()
        if self.precision == "double":
            return x.double()


def get_dim(name):
    if name == "voc":
        return 3 * 224 * 224

def get_num_labels(name):
    if name == "voc":
        return 20

def get_normalization_shape(name):
    if name == "voc":
        return (3, 1, 1)

def get_normalization_stats(name):
    if name == "voc":
        return {"mu": [0.485, 0.456, 0.406], "sigma": [0.229, 0.224, 0.225]}

def get_dataset(name, split, precision):

    precision_transform = PrecisionTransform(precision)

    if name == "voc" and split == "train":
        return datasets.VOCDetection("./data/voc", image_set="train" download=True,
                                     transform=transforms.Compose([
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   precision_transform]))

    if name == "voc" and split == "test" or split =="val":
        return datasets.VOCDetection("./data/voc", image_set=split, download=True,
                                     transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   precision_transform]))

    raise ValueError

