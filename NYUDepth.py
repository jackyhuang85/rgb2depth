import os
import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from skimage.transform import resize, rescale, rotate
from scipy.io import loadmat


class NYUDepth(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): path to .mat folder
            mode (string): 'train' or 'test'
            transforms (callable, optional): transforms on samples
        """
        self.root = root
        self.mode = mode

        self.nyu_path = os.path.join(self.root, 'nyu_depth_v2_labeled.mat')
        self.split_path = os.path.join(self.root, 'splits.mat')

        self.nyu = h5py.File(self.nyu_path)
        # ['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances',
        #  'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths',
        #  'rawRgbFilenames', 'sceneTypes', 'scenes']
        # (1499, 3, 640, 480)
        self.image = self.nyu['images']
        # (1499, 640, 480)
        self.depth = self.nyu['depths']
        # train:795, test:654
        self.split = loadmat(self.split_path)

        if self.mode == 'train':
            self.list = self.split['trainNdxs'][:, 0]
        elif self.mode == 'test':
            self.list = self.split['testNdxs'][:, 0]
        else:
            raise ValueError('mode should be \'train\' or \'test\'')
        # import pdb
        # pdb.set_trace()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        i = self.list[index]-1
        image = self.image[i].transpose(1, 2, 0)
        depth = self.depth[i]

        if self.mode == 'train':
            image, depth = self.train_transform(image, depth)
        else:
            image, depth = self.test_transform(image, depth)
        # import pdb
        # pdb.set_trace()
        return image, depth.copy()

    def train_transform(self, img, depth):
        scale = np.random.uniform(1.0, 1.5)
        angle = np.random.uniform(-5.0, 5.0)
        flip = np.random.uniform(0.0, 1.0) < 0.5

        transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(250)
        ])
        # then rotate
        transform2 = transforms.Compose([
            transforms.Resize(int(240*scale)),
            transforms.CenterCrop((304, 228))
        ])
        # then flip
        transform3 = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor()
        ])

        img = transform1(img)
        img = transforms.functional.rotate(img, angle)
        img = transform2(img)
        if flip:
            img = transforms.functional.hflip(img)
        img = transform3(img)

        # import pdb
        # pdb.set_trace()
        size1 = (int(640*250/480), 250)
        depth = depth / scale
        depth = resize(depth, size1, preserve_range=True)
        depth = rotate(depth, angle, preserve_range=True)
        depth = rescale(depth, scale, preserve_range=True)
        # center crop
        h, w = depth.shape
        h, w = (h-304)//2, (w-228)//2
        depth = depth[h:h+304, w:w+228]
        if flip:
            depth = depth[::-1, :]

        return img, depth

    def test_transform(self, img, depth):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(240),
            transforms.CenterCrop((304, 228)),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor()
        ])
        img = transform(img)

        size1 = (320, 240)
        depth = resize(depth, size1, preserve_range=True)
        # center crop
        h, w = depth.shape
        h, w = (h-304)//2, (w-228)//2
        depth = depth[h:h+304, w:w+228]
        return img, depth
