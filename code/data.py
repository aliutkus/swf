# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.multiprocessing as mp
import time
from math import floor
from celeba import CelebA
import os
import atexit
from functools import partial
from contextlib import contextmanager


class DataStream:
    "A DataStream object puts items from a Dataset into a queue"

    def __init__(self,
                 dataset,
                 maxsize=0, num_epochs=-1, queue=None):
        """creates a new datastream object. If num_epoch is negative, will
        loop endlessly. If the queue object is None, will create a new one"""

        # go into a start method that works with pytorch queues
        self.ctx = mp.get_context('spawn')

        # Allocate the data queue
        self.queue = self.ctx.Queue(maxsize=30)
        self.manager = self.ctx.Manager()

        # prepare some data for the synchronization of the workers
        self.params = self.manager.dict()
        self.params['dataset'] = dataset
        self.params['die'] = False
        self.params['num_epochs'] = num_epochs

        # create a lock
        self.lock = self.ctx.Lock()

    def start(self):
        self.process = self.ctx.Process(
                            target=data_worker,
                            kwargs={'lock': self.lock,
                                    'params': self.params,
                                    'data_queue': self.queue})
        atexit.register(partial(exit_handler, stream=self))
        self.process.start()


def exit_handler(stream):
    print('Terminating data worker...')
    if stream.params is not None:
        stream.params['die'] = True
    stream.process.join()
    print('done')


def data_worker(lock, params, data_queue):
    @contextmanager
    def getlock():
        # get the lock of the stream to manipulate the stream.data
        result = lock.acquire(block=True)
        yield result
        if result:
            lock.release()

    epoch = 0
    with getlock():
        dataset = params['dataset']
        num_epochs = params['num_epochs']

    use_cuda = False
    if use_cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        num_workers = 0#max(1, floor((mp.cpu_count()-1)/2))
        kwargs = {'num_workers': num_workers}
    data_source = DataLoader(dataset, batch_size=600, **kwargs)

    print('Starting the DataStream worker')
    while num_epochs < 0 or epoch < num_epochs:
        check = 100
        for (X, Y) in data_source:
            data_queue.put((X, Y))
            check -= 1
            if check == 0:
                check = 100
                with getlock():
                    if params['die']:
                        break
        epoch += 1


def load_image_dataset(dataset, data_dir="data", img_size=None, mode='train'):
    """handles torchvision datasets and celebA"""

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
    return data


class FnDataset(Dataset):
    def __init__(self, dataset, function):
        self.dataset = dataset
        self.function = function

    def __getitem__(self, index):
        X, y = self.dataset[index]
        return (self.function(X), y)

    def __len__(self):
        return len(self.dataset)


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
