import os
import numpy as np
import pandas as pd
import torch

from loader.custom import CustomDataset
from utils.config import Config

config = Config()


class ZILIONIS(CustomDataset):
    """
    Load the Zilionis dataset
    """

    def __init__(self, dir_path=None, n_samples=0, *args, **kwargs):
        dir_path = os.path.join(config["data_path"], "ZILIONIS")
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(n_samples=n_samples, *args, **kwargs)

    def create(self):
        """
        Generate a figure-8 dataset.
        """

        pca306 = pd.read_csv(
            os.path.join(self.dir_path, "cancer_qc_final.txt"), sep="\t", header=None
        )
        pca306 = torch.tensor(pca306.to_numpy())

        meta = pd.read_csv(
            os.path.join(self.dir_path, "cancer_qc_final_metadata.txt"),
            sep="\t",
            header=0,
        )

        cell_types = meta["Major cell type"].to_numpy()

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)
        pca306 = pca306.float()
        labels = labels.float()

        mean_dataset = torch.mean(pca306, dim=1)
        std_dataset = torch.std(pca306, dim=1)
        pca306 = (pca306 - mean_dataset[:, None]) / std_dataset[:, None]

        return pca306, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(
            os.path.join(dir_path, "cancer_qc_final_metadata.txt"), sep="\t", header=0
        )
        cell_types = meta["Major cell type"].to_numpy()

        string_labels = np.unique(cell_types)
        string_labels = np.array([cell_type[1:] for cell_type in string_labels])

        return list(string_labels)
