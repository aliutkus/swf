# imports
import numpy as np
import torch
from torch.utils.data import Dataset
from percentile import Percentile
import atexit
import queue
import torch.multiprocessing as mp
from contextlib import contextmanager
from functools import partial


class Projectors:
    """Each projector is a set of unit-length random vector"""

    def __init__(self, num_thetas, data_shape):
        self.num_thetas = num_thetas
        self.data_shape = data_shape
        self.data_dim = np.prod(np.array(data_shape))
        # for now, always use the CPU for generating projectors
        self.device = "cpu"

    def __getitem__(self, idx):
        device = torch.device(self.device)

        if isinstance(idx, int):
            idx = [idx]
        result = torch.empty((len(idx), self.num_thetas, self.data_dim),
                             device=device)
        for pos, id in enumerate(idx):
            torch.manual_seed(id)
            result[pos] = torch.randn(self.num_thetas, self.data_dim,
                                      device=device)
            #result[pos][torch.abs(result[pos]) < 0.2] = 0
            result[pos] /= (torch.norm(result[pos], dim=1, keepdim=True))
        return torch.squeeze(result)


class Sketcher(Dataset):
    """Sketcher class: takes a source of data, a dataset of projectors, and
    construct sketches.
    When accessing one of its elements, computes the corresponding sketch.
    When iterated upon, computes random batches of sketches.
    """
    def __init__(self,
                 dataloader,
                 projectors,
                 num_quantiles,
                 seed=0):
        self.data_source = dataloader
        self.projectors = projectors
        self.num_quantiles = num_quantiles

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

    def __getitem__(self, index):
        # get the device
        device = torch.device(self.device)

        # get the projector
        projector = self.projectors[index].view([-1, self.projectors.data_dim])
        projector = projector.to(device)

        # allocate the projectons variable
        projections = torch.empty((projector.shape[0],
                                   len(self.data_source.sampler)),
                                  device=device)

        # compute the projections by a loop over the data
        pos = 0
        for imgs, labels in self.data_source:
            # get a batch of images and send it to device
            imgs = imgs.to(device)

            # if required, flatten each sample
            # if imgs.shape[-1] != self.projectors.data_dim:
            imgs = imgs.view([-1, self.projectors.data_dim])

            # aggregate the projections
            projections[:, pos:pos+len(imgs)] = \
                torch.mm(projector, imgs.transpose(0, 1))
            pos += len(imgs)

        # compute the quantiles for these projections
        return (Percentile(self.num_quantiles, device)(projections).float(),
                projector, index)


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
        import time
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

    def start(self, num_workers, num_sketches,
              dataloader, projectors, num_quantiles):
        # first stop if it was started before
        self.stop()

        # now create the queue. If infinite, then set an arbitrary maxsize
        self.queue = self.ctx.Queue(maxsize=10 if num_sketches < 0
                                    else num_sketches)

        self.manager = self.ctx.Manager()
        self.data = self.manager.dict()
        self.data['in_progress'] = 0
        self.data['max_counter'] = (num_sketches if num_sketches > 0
                                    else np.inf)
        self.data['counter'] = 0
        self.lock = self.ctx.Lock()
        if num_workers < 0:
            num_workers = np.inf
        num_workers = max(1, min(num_workers,
                                 int((mp.cpu_count()-2)/2)))

        self.processes = [self.ctx.Process(target=sketch_worker,
                                           kwargs={'sketcher':
                                                   Sketcher(dataloader,
                                                            projectors,
                                                            num_quantiles),
                                                   'stream': self})
                          for n in range(num_workers)]

        atexit.register(partial(exit_handler, stream=self))

        for p in self.processes:
            p.start()

    def pause(self):
        if self.data is None:
            return
        self.data['pause'] = True
        while self.data['in_progress'] > 0:
            pass

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
        result = stream.lock.acquire(timeout=1)
        yield result
        if result:
            stream.lock.release()

    pause_displayed = False
    while True:
        id_obtained = False
        if ('pause' not in stream.data) or (not stream.data['pause']):
            if pause_displayed:
                print('Sketch worker back from sleep')
                pause_displayed = False

            with getlock():
                id = stream.data['counter']
                if id >= stream.data['max_counter']:
                    # they can die.
                    # we reached the limit, we let the other workers know
                    print("Obtained id %d is over the target number of "
                          "sketches. Pausing sketching " % id)
                    stream.data['pause'] = True
                else:
                    id_obtained = True
                    stream.data['counter'] += 1
                    stream.data['in_progress'] += 1

        if id_obtained:
            (target_qf, projector, id) = sketcher[id]
            stream.queue.put(((target_qf, projector, id)))
            with stream.lock:
                stream.data['in_progress'] -= 1

        if 'die' in stream.data:
            print('Sketch worker dying')
            break

        if 'pause' in stream.data and stream.data['pause']:
            if not pause_displayed:
                print('Sketch worker going to sleep')
                pause_displayed = True
            import time
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
                        help="Number of sketches to compute. If negative, "
                             "take an infinite number of them.",
                        type=int,
                        default=-1)
    parser.add_argument("--num_workers",
                        help="Number of workers. If not provided, use"
                        " all CPUs",
                        type=int,
                        default=-1)

    return parser
