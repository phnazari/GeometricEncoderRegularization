from torch.utils import data

from loader.MNIST_dataset import MNIST
from loader.EARTH_dataset import EARTH
from loader.POLSURF_dataset import POLSURF
from loader.ZILIONIS_dataset import ZILIONIS
from loader.CELEGANS_dataset import CELEGANS
from loader.PBMC_dataset import PBMC

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader

def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == 'MNIST':
        dataset = MNIST(**data_dict)
    elif name == "EARTH":
        dataset = EARTH(**data_dict)
    elif name == "POLSURF":
        dataset = POLSURF(**data_dict)
    elif name == "ZILIONIS":
        dataset = ZILIONIS(**data_dict)
    elif name == "CELEGANS":
        dataset = CELEGANS(**data_dict)
    elif name == "PBMC":
        dataset = PBMC(**data_dict)
    return dataset