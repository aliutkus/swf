# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.multiprocessing as mp
from math import floor
from celeba import CelebA
import os


class DynamicSubsetRandomSampler(Sampler):
    r"""Samples a given number of elements randomly, with replacement.

    Arguments:
        data_source (Dataset): elements to sample from
        nb_items (int): number of samples to draw each time
    """

    def __init__(self, data_source, nb_items):
        self.data_source = data_source
        self.nb_items = nb_items

    def __iter__(self):
        return iter(list(np.random.randint(low=0, high=len(self.data_source),
                                           size=self.nb_items)))

    def __len__(self):
        return self.nb_items


def load_data(dataset, data_dir="data", img_size=None,
              clipto=None, batch_size=640, use_cuda=False):
    if use_cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        num_workers = max(1, floor((mp.cpu_count()-2)/2))
        kwargs = {'num_workers': num_workers}

    # First load the DataSet
    if os.path.isfile(dataset):
        # this is a file, and hence should be a ndarray saved by numpy.save
        imgs = torch.tensor(np.load(dataset))
        data = TensorDataset(imgs, torch.zeros(imgs.shape[0]))
    else:
        # this is not a file. In this case, there will be a DataSet class to
        # handle it.

        # first define the transforms
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        # If it's a dir and is celebA, then we have a special loader
        if os.path.isdir(dataset):
            if os.path.basename(dataset).upper() == "CELEBA":
                data = CelebA(dataset, transform, mode="train")
        else:
            # Just assume it's a torchvision dataset
            DATASET = getattr(datasets, dataset)
            data = DATASET(data_dir, train=True, download=True,
                           transform=transform)

    # Now get a dataloader
    nb_items = len(data) if clipto < 0 else clipto
    sampler = DynamicSubsetRandomSampler(data, nb_items)
    data_loader = DataLoader(data,
                             sampler=sampler,
                             batch_size=min(nb_items, batch_size),
                             **kwargs)
    return data_loader


def add_data_arguments(parser):
    parser.add_argument("dataset",
                        help="either: i/ a file saved by numpy.save "
                             "containing a ndarray of shape "
                             "(num_samples,)+data_shape, or ii/ the path "
                             "to the celebA directory (that must end with"
                             "`CelebA` or iii/ the name of one of the "
                             "datasets in torchvision.datasets")
    parser.add_argument("--img_size",
                        help="Images are resized as s x s",
                        type=int,
                        default=64)
    parser.add_argument("--root_data_dir",
                        help="Root directory of the dataset. Defaults to"
                             "`data/\{dataset\}`")
    return parser
