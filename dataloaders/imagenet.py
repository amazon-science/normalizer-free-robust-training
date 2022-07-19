'''
Adapted from these sources:
https://github.com/snu-mllab/PuzzleMix/blob/master/load_data.py with MIT license
https://github.com/google-research/augmix with Apache-2.0 license
https://github.com/hendrycks/imagenet-r with MIT license
https://github.com/clovaai/CutMix-PyTorch with MIT license
https://github.com/LiYingwei/ShapeTextureDebiasedTraining with MIT license
'''

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import numpy as np 
import os, shutil

from utils.transforms import ColorJitter, Lighting

def imagenet_datasets(data_dir, min_crop_scale=0.08, num_classes=1000):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    validation_root = os.path.join(data_dir, 'val')  # this is path to validation images folder
    print('Training images loading from %s' % train_root)
    print('Validation images loading from %s' % validation_root)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224, scale=(min_crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        preprocess]) 
    test_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess])

    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(validation_root, transform=test_transform)

    if num_classes > 0 and num_classes < 1000:
        selected_classes = np.arange(num_classes)
        indices = [i for i, label in enumerate(train_data.targets) if label in selected_classes]
        train_data = Subset(train_data, indices)
        indices = [i for i, label in enumerate(test_data.targets) if label in selected_classes]
        test_data = Subset(test_data, indices)
        print('ImageNet-%d train %d test %d' % (num_classes, len(train_data), len(test_data)))
    
    return train_data, test_data

def imagenet_deepaug_dataset(data_dir, min_crop_scale=0.08, num_classes=1000):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    print('Training images loading from %s' % train_root)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224, scale=(min_crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        preprocess]) 

    train_data = datasets.ImageFolder(train_root, transform=train_transform)

    if num_classes > 0 and num_classes < 1000:
        selected_classes = np.arange(num_classes)
        indices = [i for i, label in enumerate(train_data.targets) if label in selected_classes]
        train_data = Subset(train_data, indices)
        print('ImageNet-DeepAug-%d train %d' % (num_classes, len(train_data)))
    
    return train_data

## Texture debias augmentation training dataset
def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

def imagenet_texture_debias_dataset(data_dir, min_crop_scale=0.64, num_classes=1000):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_root = os.path.join(data_dir, 'train')  # this is path to training images folder
    print('Training images loading from %s (with texture debiased augmentations)' % train_root)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224, scale=(min_crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([get_color_distortion()], p=0.5),
        preprocess]) 

    train_data = datasets.ImageFolder(train_root, transform=train_transform)

    if num_classes > 0 and num_classes < 1000:
        selected_classes = np.arange(num_classes)
        indices = [i for i, label in enumerate(train_data.targets) if label in selected_classes]
        train_data = Subset(train_data, indices)
        print('ImageNet-DeepAug-%d train %d' % (num_classes, len(train_data)))
    
    return train_data

def imagenet_sketch_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print('Sketch images loading from %s' % data_dir)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess])

    test_data = datasets.ImageFolder(data_dir, transform=test_transform)
    
    return test_data

def imagenet_r_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print('ImageNet-R images loading from %s' % data_dir)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess])

    test_data = datasets.ImageFolder(data_dir, transform=test_transform)
    
    return test_data

def imagenet_a_dataset(data_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print('ImageNet-A images loading from %s' % data_dir)
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess])

    test_data = datasets.ImageFolder(data_dir, transform=test_transform)
    
    return test_data

def imagenet_c_dataset(corruption, severity, data_dir='/ssd1/haotao/ImageNet-C'):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess])
    test_root = os.path.join(data_dir, corruption, str(severity))
    test_data = datasets.ImageFolder(test_root, transform=test_transform)

    return test_data

if __name__ == '__main__':
    _, val_data = imagenet_datasets(data_dir='/home/haotao/datasets/imagenet', num_classes=100)
    # train_data, _ = imagenet_datasets(data_dir='/home/haotao/datasets/imagenet')
    # train_loader = DataLoader(train_data, batch_size=10, shuffle=False, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False, num_workers=1, pin_memory=False)
    for i, (images, labels) in enumerate(val_loader):
        print(images.shape)
        print(labels.shape)
