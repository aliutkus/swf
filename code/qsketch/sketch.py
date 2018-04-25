# imports
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import tqdm
import argparse


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
            #result[pos] = np.random.randn(self.data_dim, self.data_dim)
            #result[pos] /= np.linalg.norm(result[pos], axis=-1)

            '''A Random matrix distributed with Haar measure,
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
            result[pos] = np.multiply(q, ph, q)
        return np.squeeze(result)


def main_sketch(dataset, output, num_sketches,
                num_quantiles, img_size,
                memory_usage, data_dir, clipto=None):

    # Data loading
    DATASET = getattr(datasets, dataset)
    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()])
    data = DATASET(data_dir, train=True, download=True,
                   transform=transform)
    data_dim = int(np.prod(data[0][0].size()))
    nchannels = data[0][0].shape[0]

    # computing batch size, that fits in memory
    data_bytes = data_dim * data[0][0].element_size()
    nimg_batch = int(memory_usage*2**30 / data_bytes)
    num_samples = len(data)
    if clipto is not None:
        num_samples = min(num_samples, clipto)
        nimg_batch = min(nimg_batch, clipto)
    nimg_batch = min(nimg_batch, num_samples)

    data_loader = torch.utils.data.DataLoader(data, batch_size=nimg_batch)

    # prepare the sketch data
    projectors = Projectors(data_dim=data_dim, size=num_sketches)
    sketch_loader = DataLoader(range(num_sketches))

    print('Sketching the data')
    quantiles = np.linspace(0, 100, num_quantiles)
    # allocate the sketch variable (quantile function)
    qf = np.zeros((num_sketches, data_dim, num_quantiles))

    if nimg_batch == num_samples:
        # Everything fits into memory: load only once
        print('Loading all data...')
        for img, labels in data_loader:
            imgs_all = torch.Tensor(img).view(-1, data_dim).numpy()
        print('done.')
    else:
        # Will have to load several times
        imgs_all = None

    # proceed to projection
    for batch in tqdm.tqdm(sketch_loader):
        # initialize projections
        batch_proj = projectors[batch]
        if imgs_all is None:
            # loop over the data if not loaded once
            pos = 0
            projections = np.zeros((data_dim, num_samples))
            for img, labels in tqdm.tqdm(data_loader):
                # load img numpy data
                imgs_npy = torch.Tensor(img).view(-1, data_dim).numpy()
                projections[:, pos:pos+len(img)] = batch_proj.dot(imgs_npy.T)
                pos += len(img)
        else:
            projections = batch_proj.dot(imgs_all.T)

        # compute the quantiles for each of these projections
        qf[batch] = np.percentile(projections, quantiles, axis=1).T

    # save sketch
    np.save(output, {'qf': qf, 'img_size': img_size, 'nchannels': nchannels})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Sketch a torchvision dataset.')
    parser.add_argument("dataset", help="the name of the dataset to sketch, "
                        "must be one of those supported in "
                        "torchvision.datasets")
    parser.add_argument("-n", "--num_sketches",
                        help="Number of sketches",
                        type=int,
                        default=400)
    parser.add_argument("-s", "--img_size",
                        help="Images are resized as sxs",
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
