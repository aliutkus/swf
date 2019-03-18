# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.multiprocessing as mp
import time
from math import floor
from celeba import CelebA
import os


class DataStream:
    "A DataStream object puts items from a Dataset into a queue"

    def __init__(self, data_source, maxsize=0, num_epochs=-1, queue=None):
        """creates a new datastream object. If num_epoch is negative, will
        loop endlessly. If the queue object is None, will create a new one"""
        self.data_source = data_source
        self.num_epochs = num_epochs

        # go into a start method that works with pytorch queues
        self.ctx = mp.get_context('fork')

        # Allocate the data queue
        self.queue = self.ctx.Queue(maxsize=30)

        self.data = {'pause': False,
                     'die': False}

    def start(self):
        self.process = self.ctx.Process(target=data_worker,
                                        kwargs={'stream': self})
        self.process.start()


def exit_handler(stream):
    print('Terminating data worker...')
    if stream.data is not None:
        stream.data['die'] = True
    stream.process.join()
    print('done')


def data_worker(stream):
    epoch = 0
    print('starting the data_worker')
    while stream.num_epochs < 0 or epoch < stream.num_epochs:
        print('Data stream: epoch %d' % epoch)
        if not stream.data['pause']:
            for (X, Y) in stream.data_source:
                stream.queue.put((X, Y))
                if stream.data['die']:
                    break
                while stream.data['pause']:
                    time.sleep(5)
            epoch += 1


def load_data(dataset, data_dir="data", img_size=None,
              clipto=None, batch_size=600, use_cuda=False, mode='train'):
    if use_cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        num_workers = max(1, floor((mp.cpu_count()-1)/2))
        kwargs = {'num_workers': num_workers}

    # First load the DataSet
    if os.path.isfile(dataset):
        # this is a file, and hence should be a ndarray saved by numpy.save
        if mode == "test":
            print('WARNING: unless you specified another file, this dataset '
                  'will be the same in train and test mode')
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

        # If it's celebA, then we have a special loader
        if os.path.basename(dataset).upper() == "CELEBA":
            data = CelebA(data_dir, transform, mode=mode)
        else:
            # Just assume it's a torchvision dataset
            DATASET = getattr(datasets, dataset)
            data = DATASET(data_dir, train=(mode == 'train'),
                           download=True,
                           transform=transform)

    data_loader = DataLoader(data, batch_size=batch_size, **kwargs)
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
