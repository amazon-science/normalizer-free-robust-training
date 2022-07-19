'''
Run this script under deepaug folder to visualize some deep augmented TIN images.
'''

import numpy as np 
import argparse, sys
sys.path.append('../')
from skimage.io import imsave

from dataloaders.tiny_imagenet import tiny_imagenet_dataloaders
from utils.utils import fourD2threeD

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '--dd', default='/ssd2/haotao/tiny-imagenet-200')
args = parser.parse_args()

# deepaug:
deepaug_train_loader, test_loader = tiny_imagenet_dataloaders(args.data_dir, train_batch_size=16, transform_train=False, shuffle_train=False, deepaug=True)

imgs, labels = next(iter(deepaug_train_loader))
imgs = imgs.cpu().numpy()
print('imgs:', imgs.shape)
imgs = np.swapaxes(imgs, 1, -1)
print('imgs:', imgs.shape)

deepaug_big_img = fourD2threeD(imgs, n_row=4)
imsave('deepaug_tin_samples.png', deepaug_big_img)

# normal:
normal_train_loader, test_loader = tiny_imagenet_dataloaders(args.data_dir, train_batch_size=16, transform_train=False, shuffle_train=False, deepaug=False)

imgs, labels = next(iter(normal_train_loader))
imgs = imgs.cpu().numpy()
print('imgs:', imgs.shape)
imgs = np.swapaxes(imgs, 1, -1)
print('imgs:', imgs.shape)

normal_big_img = fourD2threeD(imgs, n_row=4)
imsave('normal_tin_samples.png', normal_big_img)


# delta:
delta = np.mean((normal_big_img[:,:,:] - deepaug_big_img[:,:,:])**2)**0.5
print(delta)
