from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import Dataset

from utils.utils import minmax, cmap_labels

class CustomDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, n_samples=0, split=None, **kwargs):
        """
        Base Class for a Custom Dataset
        """

        super().__init__()

        # if n_samples != 0:
        self.n_samples = n_samples

        self.dataset, self.coordinates = self.create()

        if len(torch.unique(self.coordinates)) > 1:
            self.labels = self.minmax(self.coordinates)
        else:
            self.labels = self.coordinates

        self.labels = self.coordinates

        # truncate dataset
        if n_samples != 0:
            test_size = int(0.1 * n_samples)
            train_size = n_samples - test_size
        else:
            test_size = int(0.1 * len(self.dataset))
            train_size = len(self.dataset) - test_size

        rest = len(self.dataset) - test_size - train_size

        train_subset, test_subset, _ = torch.utils.data.random_split(self.dataset,
                                                                     [train_size, test_size, rest],
                                                                     generator=torch.Generator().manual_seed(42))

        if split == "training":
            self.n_samples = train_size
            self.dataset = self.dataset[train_subset.indices]
            self.labels = self.labels[train_subset.indices]
        elif split == "all":
            self.n_samples = len(self.dataset)
        else:
            self.n_samples = test_size
            self.dataset = self.dataset[test_subset.indices]
            self.labels = self.labels[test_subset.indices]

        self.data = self.dataset
        self.targets = self.labels


    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        item = self.dataset[index]
        label = self.labels[index]

        return item, label

    @staticmethod
    def transform_labels(labels):
        return cmap_labels(labels)

    @staticmethod
    def minmax(item):
        return minmax(item)

    @abstractmethod
    def create(self):
        """
        Create the dataset
        """