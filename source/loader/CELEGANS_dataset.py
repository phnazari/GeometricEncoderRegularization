import os
import numpy as np
import pandas as pd
import torch

from loader.custom import CustomDataset
from utils.config import Config

config = Config()


class CELEGANS(CustomDataset):
    """
    Load the C-Elegans dataset
    """

    def __init__(self, dir_path=None, n_samples=0, *args, **kwargs):
        dir_path = os.path.join(config["data_path"], "CELEGANS")
        if dir_path is not None:
            self.dir_path = dir_path
            super().__init__(n_samples=n_samples, *args, **kwargs)

    def create(self):
        pca100 = pd.read_csv(
            os.path.join(self.dir_path, "c-elegans_qc_final.txt"), sep="\t", header=None
        )
        meta = pd.read_csv(
            os.path.join(self.dir_path, "c-elegans_qc_final_metadata.txt"),
            sep=",",
            header=0,
        )

        # remove instances where celltype is unknown
        meta["cell.type"] = meta["cell.type"].fillna("unknown")

        pca100 = torch.tensor(pca100.to_numpy())
        cell_types = meta["cell.type"].to_numpy()

        labels = np.zeros(len(cell_types)).astype(int)
        for i, phase in enumerate(np.unique(cell_types)):
            labels[cell_types == phase] = i

        labels = torch.tensor(labels)

        pca100 = pca100.float()
        labels = labels.float()

        return pca100, labels

    @staticmethod
    def transform_labels(dir_path):
        meta = pd.read_csv(
            os.path.join(dir_path, "c-elegans_qc_final_metadata.txt"), sep=",", header=0
        )

        # remove instances where celltype is unknown
        meta["cell.type"] = meta["cell.type"].fillna("unknown")
        cell_types = meta["cell.type"].to_numpy()

        string_labels = np.unique(cell_types)
        return list(string_labels)
