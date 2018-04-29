# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm
import argparse
import os


class Projectors(Dataset):
    def __init__(self, data_dim=4096, size=100):
        super(Dataset, self).__init__()
        self.data_dim = data_dim
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        result = np.empty((len(idx), self.data_dim, self.data_dim))
        for pos, id in enumerate(idx):
            np.random.seed(id)
            result[pos] = np.random.randn(self.data_dim, self.data_dim)
            result[pos] /= self.data_dim
            #result[pos] /= np.linalg.norm(result[pos], axis=-1)*self.data_dim

            """'''A Random matrix distributed with Haar measure,
            From Francesco Mezzadri:
            @article{mezzadri2006generate,
                title={How to generate random matrices from the
                classical compact groups},
                author={Mezzadri, Francesco},
                journal={arXiv preprint math-ph/0609050},
                year={2006}}
            '''
            z = np.random.randn(self.data_dim, self.data_dim)
            q, r = np.linalg.qr(z)
            d = np.diagonal(r)
            ph = d/np.absolute(d)
            result[pos] = np.multiply(q, ph, q)"""
        return np.squeeze(result)


def load_data(dataset, clipto,
              data_dir="data", img_size=None, memory_usage=2):
    # Data loading
    if os.path.exists(dataset):
        # this is a ndarray saved by numpy.save. Just dataset and clipto are
        # useful
        imgs_npy = np.load(dataset)
        (num_samples, data_dim) = imgs_npy.shape
        if clipto is not None:
            imgs_npy = imgs_npy[:min(num_samples, clipto)]
            (num_samples, data_dim) = imgs_npy.shape
        data_loader = None
    else:
        # this is a torchvision dataset
        DATASET = getattr(datasets, dataset)
        transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])
        data = DATASET(data_dir, train=True, download=True,
                       transform=transform)
        data_dim = int(np.prod(data[0][0].size()))

        # computing batch size, that fits in memory
        data_bytes = data_dim * data[0][0].element_size()
        nimg_batch = int(memory_usage*2**30 / data_bytes)
        num_samples = len(data)
        if clipto is not None:
            num_samples = min(num_samples, clipto)
            nimg_batch = min(nimg_batch, clipto)
        nimg_batch = min(nimg_batch, num_samples)

        data_loader = torch.utils.data.DataLoader(data, batch_size=nimg_batch)

        if nimg_batch == num_samples:
            # Everything fits into memory: load only once
            for img, labels in data_loader:
                imgs_npy = torch.Tensor(img).view(-1, data_dim).numpy()
        else:
            # Will have to load several times
            imgs_npy = None
    return imgs_npy, data_loader, num_samples, data_dim


def main_sketch(dataset, output, num_sketches,
                num_quantiles, img_size,
                memory_usage, data_dir, clipto=None):

    # load data
    (imgs_npy, data_loader,
     num_samples, data_dim) = load_data(dataset, clipto, data_dir,
                                        img_size, memory_usage)

    # prepare the sketch data
    projectors = Projectors(data_dim=data_dim, size=num_sketches)
    sketch_loader = DataLoader(range(num_sketches))

    print('Sketching the data')
    # allocate the sketch variable (quantile function)
    quantiles = np.linspace(0, 100, num_quantiles)
    qf = np.zeros((num_sketches, data_dim, num_quantiles))

    # proceed to projection
    for batch in tqdm.tqdm(sketch_loader):
        # initialize projections
        batch_proj = projectors[batch]
        if imgs_npy is None:
            # loop over the data if not loaded once (torchvision dataset)
            pos = 0
            projections = np.zeros((data_dim, num_samples))
            for img, labels in tqdm.tqdm(data_loader):
                # load img numpy data
                imgs_npy = torch.Tensor(img).view(-1, data_dim).numpy()
                projections[:, pos:pos+len(img)] = batch_proj.dot(imgs_npy.T)
                pos += len(img)
        else:
            # data is in memory as a ndarray
            projections = batch_proj.dot(imgs_npy.T)

        # compute the quantiles for each of these projections
        qf[batch] = np.percentile(projections, quantiles, axis=1).T

    # save sketch
    np.save(output, {'qf': qf, 'data_dim': data_dim})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Sketch a torchvision dataset or a '
                                     'ndarray saved on disk.')
    parser.add_argument("dataset", help="either a file saved by numpy.save "
                                        "containing a ndarray of shape "
                                        "num_samples x data_dim, or the name "
                                        "of the dataset to sketch, which "
                                        "must be one of those supported in "
                                        "torchvision.datasets")
    parser.add_argument("-n", "--num_sketches",
                        help="Number of sketches",
                        type=int,
                        default=400)
    parser.add_argument("-s", "--img_size",
                        help="Images are resized as s x s",
                        type=int,
                        default=64)
    parser.add_argument("-q", "--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
    parser.add_argument("-m", "--memory_usage",
                        help="RAM usage for batches in Gb",
                        type=int,
                        default=2)
    parser.add_argument("-r", "--root_data_dir",
                        help="Root directory of the dataset. Defaults to"
                             "`data/\{dataset\}`")
    parser.add_argument("-o", "--output",
                        help="Output file, defaults to the `dataset` argument")

    args = parser.parse_args()
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset
    if args.output is None:
        args.output = args.dataset
    main_sketch(args.dataset,
                args.output,
                args.num_sketches,
                args.num_quantiles, args.img_size,
                args.memory_usage, args.root_data_dir)
