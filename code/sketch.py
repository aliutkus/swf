# imports
import numpy as np
import torch
from torch.utils.data import Dataset
from percentile import Percentile
import atexit
import queue
import torch.multiprocessing as mp
from torch import nn
from contextlib import contextmanager
from functools import partial
import time


class LinearWithBackward(nn.Linear):
    def __init__(self, **kwargs):
        super(LinearWithBackward, self).__init__(**kwargs)
        self.weight = torch.nn.Parameter(
            self.weight/torch.norm(self.weight, dim=1, keepdim=True))

    def forward(self, input):
        return super(LinearWithBackward, self).forward(
            input.view(input.shape[0], -1))

    def backward(self, grad):
        return torch.mm(grad.view(grad.shape[0], -1), self.weight)


class Projectors:
    """Each projector is a set of unit-length random vector"""

    def __init__(self, num_thetas, data_shape):
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))
        # for now, always use the CPU for generating projectors
        self.device = "cpu"

    def __getitem__(self, indexes):
        device = torch.device(self.device)

        if isinstance(indexes, int):
            idx = [indexes]
        else:
            idx = indexes

        result = []
        for pos, id in enumerate(idx):
            torch.manual_seed(id)
            new_projector = LinearWithBackward(in_features=self.data_dim,
                                      out_features=self.num_thetas,
                                      bias=False).to(device)
            new_projector.weight = nn.Parameter(
                                    new_projector.weight
                                    / torch.norm(new_projector.weight,
                                                 dim=1, keepdim=True))
            result += [new_projector]
        return result[0] if isinstance(indexes, int) else result


class RandomCoders:
    """Each coder has random weights"""

    def __init__(self, num_thetas, data_shape):
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))
        # for now, always use the CPU for generating projectors
        self.device = "cpu"

    def __getitem__(self, indexes):
        from autoencoder import DenseEncoder

        if isinstance(indexes, int):
            idx = [indexes]
        else:
            idx = indexes

        result = []
        for pos, id in enumerate(idx):
            torch.manual_seed(id)
            print(id)
            result += [DenseEncoder(input_shape=self.data_shape,
                                    bottleneck_size=self.num_thetas)]
        return result[0] if isinstance(indexes, int) else result


class Sketcher(Dataset):
    """Sketcher class: takes a source of data, a dataset of projectors, and
    construct sketches.
    When accessing one of its elements, computes the corresponding sketch.
    When iterated upon, computes random batches of sketches.
    """
    def __init__(self,
                 data_stream,
                 projectors,
                 num_quantiles,
                 clip_to,
                 seed=0):
        self.data_source = data_stream
        self.projectors = projectors
        self.num_quantiles = num_quantiles
        self.clip_to = len(data_stream.dataset) if clip_to < 0 else clip_to

        # the random sequence should be reproductible without interfering
        # with other np random seed stuff. To achieve this, we
        # backup current numpy random state, and create a new one
        # based on the provided seed. Then, we revert to the random state that
        # was there.
        random_state = np.random.get_state()
        np.random.seed(seed)
        self.random_state = np.random.get_state()
        np.random.set_state(random_state)

        # only use the cpu for now for the sketcher
        self.device = "cpu"

    def __iter__(self):
        return self

    def __next__(self):
        # saves the current numpy random state, before setting it to the last
        # one stored
        random_state = np.random.get_state()
        np.random.set_state(self.random_state)

        # generate the next id
        next_id = np.random.randint(np.iinfo(np.int32).max)

        # stores current numpy random state and restores the one we had before
        self.random_state = np.random.get_state()
        np.random.set_state(random_state)

        # finally get the sketch
        return self.__getitem__(next_id)

    def __getitem__(self, indexes):
        if isinstance(indexes, int):
            index = [indexes]
        else:
            index = indexes

        # get the device
        device = torch.device(self.device)

        # get the projector
        sketches = []
        for id in index:
            projector = self.projectors[id].to(device)

            # allocate the projectons variable
            projections = torch.empty((self.clip_to,
                                       self.projectors.num_thetas),
                                      device=device)

            # compute the projections by a loop over the data
            pos = 0
            while pos < self.clip_to:
                (imgs, labels) = self.data_source.get()
                # get a batch of images and send it to device
                imgs = imgs.to(device)

                # aggregate the projections
                projections[pos:pos+len(imgs)] = projector(imgs)
                pos += len(imgs)
            # compute the quantiles for these projections
            sketches += [
                (Percentile(
                    self.num_quantiles, device)(projections).float(),
                 projector,
                 index)]
        return sketches[0] if isinstance(indexes, int) else sketches


class SketchStream:
    """A SketchStream object constructs a queue attribute that will be
    filled with sketches, which are quantiles of random projections"""

    def __init__(self):
        self.processes = []
        self.data = None

        # go into a start method that works with pytorch queues
        self.ctx = mp.get_context('fork')

        self.in_progress = 0

        # Allocate the sketch queue
        self.queue = self.ctx.Queue()

    def dump(self, file):
        # first pausing the sketching
        self.pause()
        time.sleep(5)
        data = []
        while True:
            try:
                item = self.queue.get(block=False)
            except queue.Empty:
                break
            data += [item, ]
        with open(file, 'wb') as handle:
            torch.save(data, handle)
        self.load(file)

    def load(self, file):
        # killing the sketching in progress if any
        self.stop()

        # get the data
        with open(file, 'rb') as handle:
            data = torch.load(handle)

        # create the queue
        self.queue = self.ctx.Queue(maxsize=len(data))

        # then fill with the data
        for item in data:
            self.queue.put(item)

    def start(self, num_workers, num_epochs, num_sketches,
              data_stream, projectors, num_quantiles, clip_to):
        # first stop if it was started before
        self.stop()

        # get the number of workers
        if num_workers < 0:
            num_workers = np.inf
            num_workers = max(1, min(num_workers,
                              int((mp.cpu_count()-1)/2)))

        print('using ', num_workers, 'workers')
        # now create a queue with a maxsize corresponding to a few times
        # the number of workers
        self.queue = self.ctx.Queue(maxsize=2*num_workers)
        self.manager = self.ctx.Manager()
        self.num_epochs = num_epochs
        self.projectors = projectors
        self.data = self.manager.dict()
        self.data['pause'] = False
        self.data['current_pick_epoch'] = 0
        self.data['current_put_epoch'] = 0
        self.data['current_sketch'] = 0
        self.data['done_in_current_epoch'] = 0
        self.data['num_sketches'] = (num_sketches if num_sketches > 0
                                     else np.inf)
        self.lock = self.ctx.Lock()
        self.processes = [self.ctx.Process(target=sketch_worker,
                                           kwargs={'sketcher':
                                                   Sketcher(data_stream.queue,
                                                            projectors,
                                                            num_quantiles,
                                                            clip_to),
                                                   'stream': self})
                          for n in range(num_workers)]

        atexit.register(partial(exit_handler, stream=self))

        for p in self.processes:
            p.start()

    def pause(self):
        if self.data is None:
            return
        self.data['pause'] = True

    def restart(self):
        self.data['counter'] = 0
        self.resume()

    def resume(self):
        if self.data is not None:
            self.data['pause'] = False

    def stop(self):
        if self.data is not None:
            self.data['die'] = True


def exit_handler(stream):
    print('Terminating sketchers...')
    if stream.data is not None:
        stream.data['die'] = True
    for p in stream.processes:
        p.join()
    print('done')


def sketch_worker(sketcher, stream):
    @contextmanager
    def getlock():
        result = stream.lock.acquire(block=True)
        yield result
        if result:
            stream.lock.release()

    pause_displayed = False
    while True:
        worker_dying = False
        if not stream.data['pause']:
            if pause_displayed:
                print('Sketch worker back from sleep')
                pause_displayed = False

            #print('sketch: trying to get lock')
            with getlock():
                id = stream.data['current_sketch']
                epoch = stream.data['current_pick_epoch']
                #print('sketch: got lock, epoch %d and id %d' % (epoch, id))
                if epoch >= stream.num_epochs:
                    print('epoch',epoch,'greater than the number of epochs:',stream.num_epochs,'dying now')
                    worker_dying = True
                else:
                    if id == stream.data['num_sketches'] - 1:
                        # we reached the limit, we let the other workers know
                        print("Obtained id %d is last for this epoch. "
                              "Reseting the counter and incrementing current "
                              "epoch " % id)
                        stream.data['current_sketch'] = 0
                        stream.data['current_pick_epoch'] += 1
                    else:
                        stream.data['current_sketch'] += 1
            if worker_dying:
                print(id, epoch, 'Reached the desired amount of epochs. Dying.')
                while True:
                    time.sleep(10)
                return

            #print('sketch: now trying to compute id', id)
            (target_qf, projector, id) = sketcher[id]

            #print('sketch: we computed the sketch with id', id)
            while (stream.data['current_put_epoch'] != epoch):
                #print("waiting: current put epoch",stream.data['current_put_epoch'])
                pass

            #print('sketch: trying to put id',id,'epoch',epoch)
            stream.queue.put(((target_qf, projector, id)))
            #print('sketch: we put id', id, 'epoch', epoch)

            with stream.lock:
                stream.data['done_in_current_epoch'] += 1
                #print('sketch: after put, got lock. id', id, 'epoch', epoch, 'done in current epoch',stream.data['done_in_current_epoch'])
                if (
                  stream.data['done_in_current_epoch']
                  == stream.data['num_sketches']):
                    # we need to send sentinel
                    print('Sketch: sending the sentinel')
                    stream.queue.put(None)
                    stream.data['done_in_current_epoch'] = 0
                    stream.data['current_put_epoch'] += 1

        if 'die' in stream.data:
            print('Sketch worker dying')
            break

        if stream.data['pause']:
            if not pause_displayed:
                print('Sketch worker going to sleep')
                pause_displayed = True
            time.sleep(2)


def add_sketch_arguments(parser):
    parser.add_argument("--num_thetas",
                        help="Number of thetas per sketch.",
                        type=int,
                        default=2000)
    parser.add_argument("--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
    parser.add_argument("--clip_to",
                        help="Number of datapoints used per sketch. If "
                             "negative, take all of them.",
                        type=int,
                        default=3000)
    parser.add_argument("--num_sketches",
                        help="Number of sketches per epoch. If negative, "
                             "take an infinite number of them.",
                        type=int,
                        default=-1)
    parser.add_argument("--num_workers",
                        help="Number of workers for computing the sketches. "
                        "If not provided, use all CPUs",
                        type=int,
                        default=-1)

    return parser
