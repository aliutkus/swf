# imports
import os
import torch
import sketch
import data
from sketch import SketchStream
import argparse
import torch.multiprocessing as mp
from math import floor
from torchvision import transforms
from autoencoder import AE
import numpy as np

if __name__ == "__main__":
    # create arguments parser and parse arguments
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Flow.')
    parser = data.add_data_arguments(parser)

    parser.add_argument("--num_samples",
                        help="Number of samples to draw per batch to "
                             "compute the sketch of that batch",
                        type=int,
                        default=3000)
    args = parser.parse_args()

    # prepare the torch device (cuda or cpu ?)
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    # load the data
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset

    data_loader = data.load_data(args.dataset, args.root_data_dir,
                                 32, 3000, digits=[1])

    data_shape = data_loader.dataset[0][0].shape
    for X, Y in data_loader:
        print(Y)
