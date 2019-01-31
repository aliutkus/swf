# imports
import os
import torch
from torch import nn
from torch import optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import sketch
import data
from sketch import SketchStream
from percentile import Percentile
import argparse
from time import strftime, gmtime
import socket
import functools
import torch.multiprocessing as mp
import queue
from math import floor
from torchvision import transforms
from autoencoder import AE
from interp1d import Interp1d
from math import sqrt
import utils
import numpy as np
import json
from pathlib import Path
import uuid


def swmin(train_particles, test_particles, target_queue, num_quantiles,
          stepsize, regularization, num_iter,
          device_str, logger):
    """Minimizes the Sliced Wasserstein distance between a population
    and some target distribution described by a stream of marginal
    distributions over random projections."""
    # get the device
    device = torch.device(device_str)

    # prepare stuff
    criterion = nn.MSELoss()

    particles = torch.nn.Parameter(train_particles.to(device))
    optimizer = optim.Adam([particles], lr=stepsize)

    # batch index init
    index = 0

    while True:
        # get the data from the sketching queue
        target_qf, projector, id = target_queue.get()
        target_qf = target_qf.to(device)
        projector = projector.to(device)

        optimizer.zero_grad()

        # project the particles
        projections = projector(particles)

        # compute the corresponding quantiles
        percentile_fn = Percentile(num_quantiles, device)
        particles_qf = percentile_fn(projections)

        loss = criterion(particles_qf, target_qf)
        loss.backward()
        optimizer.step()

        # call the logger with the transported train and test particles
        logger({'train': particles}, index, {'train': loss})

        index += 1

        # puts back the data into the Queue if it's not full already
        if not target_queue.full():
            try:
                target_queue.put((target_qf.detach().to('cpu'),
                                  projector.to('cpu'), id),
                                 block=False)
            except queue.Full:
                pass

    return particles
