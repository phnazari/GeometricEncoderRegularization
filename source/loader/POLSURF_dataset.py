import numpy as np
import torch
from torch.utils.data import Dataset


class POLSURF(Dataset):
    def __init__(self, filename=None, **kwargs):
        self.data = torch.from_numpy(np.load(filename)).float()
        self.num_samples = self.data.shape[0]
        self.targets = torch.ones(self.data.shape[0])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return self.num_samples
