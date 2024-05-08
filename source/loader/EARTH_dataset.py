

import numpy as np
import pandas as pd
import importlib
importlib.import_module('mpl_toolkits').__path__
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset

from loader.custom import CustomDataset

class EARTH(CustomDataset):
    """
    Create an earth dataset
    """

    def __init__(self, filename=None, split="training", *args, **kwargs):
        self.filename = filename
        super().__init__(split=split, *args, **kwargs)

        # dataset contains some weird labels, which I remove here
        self.data = self.data[self.labels != 6]
        self.n_samples -= torch.sum(self.labels == 6).item()
        self.targets = self.targets[self.labels != 6]
        self.labels = self.labels[self.labels != 6]

        # print(f"EARTH split {split} | {self.data.size()}")

    @staticmethod
    def transform_labels(labels):
        string_labels = ["Africa", "Europe", "Asia", "North America", "Australia", "South America"]

        return string_labels

    def create(self):
        data = torch.load(self.filename)
        xs, ys, zs, labels = torch.unbind(data, dim=-1)
        dataset = torch.vstack((xs, ys, zs)).T.float()

        return dataset, labels

    def generate(self, n):
        """
        Generate and save the dataset
        """

        import geopandas

        bm = Basemap(projection="cyl")

        xs = []
        ys = []
        zs = []

        phis = []
        thetas = []

        # phi = long, theta = lat
        # das erste Argument is azimuth (phi), das zweite polar (theta) (in [-pi, pi])
        for phi in np.linspace(-180, 180, num=n):
            for theta in np.linspace(-90, 90, num=n):
                if bm.is_land(phi, theta):
                    phis.append(phi)
                    thetas.append(theta)

                    phi_rad = phi / 360 * 2 * np.pi
                    theta_rad = theta / 360 * 2 * np.pi

                    x = np.cos(phi_rad) * np.cos(theta_rad)
                    y = np.cos(theta_rad) * np.sin(phi_rad)
                    z = np.sin(theta_rad)

                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

        xs = torch.tensor(xs).float()
        ys = torch.tensor(ys).float()
        zs = torch.tensor(zs).float()

        # generate labels
        df = pd.DataFrame(
            {
                "longitude": phis,
                "latitude": thetas
            }
        )

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth'))
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))

        results = geopandas.sjoin(gdf, world, how="left")

        le = preprocessing.LabelEncoder()
        encoded_results = torch.tensor(le.fit_transform(results["continent"].values))

        data = torch.vstack((xs, ys, zs, encoded_results)).T

        torch.save(data, self.filename)

        return data