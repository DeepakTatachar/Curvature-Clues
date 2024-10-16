import torch
import numpy as np


class Noise(torch.utils.data.Dataset):
    def __init__(self, transform, length, mean, std, size):
        self.transform = transform
        self.target_transform = None
        self.length = length
        self.size = size

    def __len__(self):
        return self.length


class GaussianNoise(Noise):
    def __init__(self, transform=None, length=10000, mean=0, std=1, size=(3, 32, 32)):
        super(GaussianNoise, self).__init__(transform, length, mean, std, size)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = np.random.normal(0, 1, self.size).astype(np.float32)
        target = -1
        return sample, target


class UniformNoise(Noise):
    def __init__(self, transform=None, length=10000, mean=0, std=1, size=(3, 32, 32)):
        super(UniformNoise, self).__init__(transform, length, mean, std, size)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = np.random.uniform(0, 1, self.size).astype(np.float32)
        target = -1
        return sample, target
