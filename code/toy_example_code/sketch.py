# imports
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


class Projectors:
    def __init__(self, num_thetas, data_dim):
        self.num_thetas = num_thetas
        self.data_dim = data_dim

    def __len__(self):
        return np.inf

    """Each projector is a set of unit-length random vectors"""
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        result = np.empty((len(idx), self.num_thetas, self.data_dim))
        for pos, id in enumerate(idx):
            np.random.seed(id)
            result[pos] = np.random.randn(self.num_thetas, self.data_dim)
            result[pos] /= (np.linalg.norm(result[pos], axis=1))[:, None]
        return np.squeeze(result)


def fast_percentile(V, quantiles):
    if V.shape[1] < 10000:
        return np.percentile(V, quantiles, axis=1).T
    return np.array(Parallel(n_jobs=multiprocessing.cpu_count()-1)
                    (delayed(np.percentile)(v, quantiles)
                     for v in V))


class SketchIterator:
    def __init__(self, data, projectors, num_quantiles,
                 stop=-1, clipto=-1):
        self.data = data
        self.projectors = projectors
        self.quantiles = np.linspace(0, 100, num_quantiles)
        self.stop = stop if stop >= 0 else None
        self.current = 0
        self.num_samples_per_batch = (clipto if clipto > 0
                                      else len(self.dataloader.dataset))
        self.clipto = clipto

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop is not None and (self.current >= self.stop):
            raise StopIteration
        else:
            #   import ipdb; ipdb.set_trace()
            batch_proj = self.projectors[self.current]
            self.current += 1

            (num_samples, data_dim) = self.data.shape

            # if we clip the data, randomly select a batch for this iteration
            if self.clipto is not None and self.clipto < num_samples:
                order = np.random.permutation(self.data.shape[0])
                data_proj = self.data[order[:self.clipto]]
            else:
                data_proj = self.data

            # compute the projections
            projections = batch_proj.dot(data_proj.T)

            # compute the quantiles for each of these projections
            return fast_percentile(projections, self.quantiles), batch_proj
