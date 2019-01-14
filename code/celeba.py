from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, data_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = os.path.join(data_dir,'images')
        self.attr_path = os.path.join(data_dir, 'list_attr_celeba.txt')

        self.transform = T.Compose((T.CenterCrop(178), transform))
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
            values = split[1:]

            if (i+1) < 2000:
                self.test_dataset.append(filename)
            else:
                self.train_dataset.append(filename)

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), 0

    def __len__(self):
        """Return the number of images."""
        return self.num_images
