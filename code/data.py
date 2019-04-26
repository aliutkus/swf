# imports
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms as T
from torch.utils import data
import os
import torch
import random
from PIL import Image


class TransformedDataset:
    """ Create a dataset whose items are obtained by applying a specified
    transform (function) to the items and the targets of some other dataset."""

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, indices):
        try:
            _ = iter(indices)
            iterable = True
        except TypeError as te:
            indices = [indices]
            iterable = False
        result = []
        for id in indices:
            (X, y) = self.dataset[id]
            result += [
             (X if self.transform is None else self.transform(X),
              y if self.target_transform is None else self.target_transform(y))
            ]
        return result[0] if not iterable else result

    def __len__(self):
        return len(self.dataset)


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, data_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        data_dir = os.path.join(data_dir, 'CelebA')
        self.image_dir = os.path.join(data_dir, 'images')
        self.attr_path = os.path.join(data_dir, 'list_attr_celeba.txt')

        self.transform = T.Compose([T.CenterCrop(178), transform])
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]

        # get the filenames
        lines = lines[2:]
        random.seed(0)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]

            if (i+1) < 2000:
                self.test_dataset.append(filename)
            else:
                self.train_dataset.append(filename)

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = (self.train_dataset if self.mode == 'train'
                   else self.test_dataset)
        filename = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), 0

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def load_image_dataset(dataset, data_dir="data", img_size=None, mode='train'):
    """handles torchvision datasets and celebA"""

    # first define the transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # If it's celebA, then we have a special loader
    if os.path.basename(dataset).upper() == "CELEBA":
        res_data = CelebA(data_dir, transform, mode=mode)
    elif dataset.upper() == 'TOY':
        import numpy as np
        xdata = torch.tensor(np.load('toy.npy'))
        ydata = torch.zeros(len(xdata))
        res_data = data.TensorDataset(xdata[:, None, ...], ydata)
    else:
        data_dir = os.path.join(data_dir, os.path.basename(dataset))
        # Just assume it's a torchvision dataset
        DATASET = getattr(datasets, dataset)
        res_data = DATASET(data_dir, train=(mode == 'train'),
                           download=True,
                           transform=transform)
    return res_data


def add_data_arguments(parser):
    parser.add_argument("dataset",
                        help="either: i/ the path to the celebA directory "
                             "(that must end with `CelebA` or ii/ the name of "
                             "one of the datasets in torchvision.datasets")
    parser.add_argument("--img_size",
                        help="Images are resized as s x s",
                        type=int,
                        default=64)
    parser.add_argument("--root_data_dir",
                        help="Root directory of the dataset. Defaults to"
                             "`data`",
                        default='data')
    return parser
