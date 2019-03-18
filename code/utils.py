import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from math import floor


def compare_image(image, collection, return_amount):
    lowest_mse = 1000000000
    ax = None
    arr = np.array([]).astype(np.float32)
    for i in range(len(collection)):
        mse = ((image.detach().cpu().numpy()
                - collection[i][0].detach().cpu().numpy()) ** 2).mean(axis=ax)
        arr = np.append(arr, mse)
        if mse < lowest_mse:
            lowest_mse = mse
    ind = np.argpartition(arr, return_amount)[:return_amount]
    return ind, arr


def find_closest(items, dataset):
    # bring the items to cpu and flatten them
    items = items.detach().cpu()
    num = items.shape[0]
    items = items.view(num, -1)

    # initialize the losses to infinite
    mindist = torch.ones(num)*float('inf')
    closest = torch.zeros_like(items)
    num_workers = max(1, floor((mp.cpu_count()-2)/2))
    dataloader = DataLoader(dataset, batch_size=5000, num_workers=num_workers)
    for (candidates, _) in dataloader:
        candidates = candidates.view(candidates.shape[0], -1).cpu()
        distances = torch.norm(
                        items[:, None, :] - candidates[None, ...],
                        dim=-1)
        mindist_in_batch, closest_in_batch = torch.min(distances, dim=1)
        replace = torch.nonzero(mindist_in_batch < mindist)
        closest[replace] = candidates[closest_in_batch[replace]]
    return closest
