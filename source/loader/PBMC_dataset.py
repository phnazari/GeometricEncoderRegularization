import os
import numpy as np
import pandas as pd
import torch

from loader.custom import CustomDataset
from utils.config import Config

config = Config()


class PBMC(CustomDataset):
    """
    Load the PBMC dataset
    """

    def __init__(self, dir_path=None, *args, **kwargs):
        dir_path = os.path.join(config["data_path"], "PBMC")
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(*args, **kwargs)

    def create(self):
        pca50 = np.load(os.path.join(self.dir_path, "pca50.npy"))
        pca50 = torch.from_numpy(pca50)

        meta = pd.read_csv(os.path.join(self.dir_path, "zheng17-cell-labels.txt"), sep="\t", header=None, skiprows=1)
        meta = meta.to_numpy()[:, 1]
        cell_types = np.squeeze(meta)

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)

        pca50 = pca50.float()
        labels = labels.float()

        return pca50, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(os.path.join(dir_path, "pbmc_qc_final_labels.txt"), sep="\t", header=None)
        cell_types = np.squeeze(meta.to_numpy())

        string_labels = np.unique(cell_types)

        return list(string_labels)