import torch
import numpy as np


def digitize(xp, x):
    """
    xp is d x m
    x  is d x n

    output is d x n and gives the index of the closest xp in row d less
    or equal to input.

    brute force method, not optimal at all but well."""
    distance = x[:, None, :] - xp[..., None]
    inftensor = torch.tensor(np.inf).to('cuda' if xp.is_cuda else 'cpu')
    distance = torch.where(distance < 0, inftensor, distance)
    pos = torch.argmin(distance, dim=1)
    return pos


def interpolate(xp, yp, x):
    """interpolation
    xp is dxq or 1xq
    yp is dxq or 1xq
    x is dxn

    extrapolate linearly with the closest slope available
    """
    if len(x.shape) == 1:
        x = x[None, :]
    if len(xp.shape) == 1:
        xp_flat = True
        xp = xp[None, :]
    else:
        xp_flat = False
    if len(yp.shape) == 1:
        yp_flat = True
        yp = yp[None, :]
    else:
        yp_flat = False
    # assuming xp are sorted
    slopes = (yp[:, 1:]-yp[:, :-1])/(xp[:, 1:]-xp[:, :-1])  # d x (n-1)
    num_slopes = slopes.shape[1]

    ind = digitize(xp, x)
    ind = torch.clamp(ind, 0, num_slopes - 1)

    if yp_flat:
        yp_selected = yp.view(-1)[ind]
    else:
        yp_selected = torch.gather(yp, 1, ind)
    if xp_flat:
        xp_selected = xp.view(-1)[ind]
    else:
        xp_selected = torch.gather(xp, 1, ind)

    return yp_selected + torch.gather(slopes, 1, ind)*(x-xp_selected)


if __name__ == "__main__":
    xp = torch.tensor([[0, 3, 10, 50, 100], [10, 20, 30, 40, 50]]).float()
    yp = torch.tensor([0, 25, 50, 75, 100]).float()

    x = torch.tensor([[-1.5, 1.5, 175], [-10, 30, 49]]).float()
    y = interpolate(xp, yp, x)
    print(xp, yp, x, y)

    y = torch.tensor([[10, 30], [1, 99]]).float()
    x = interpolate(yp, xp, y)
    print(xp, yp, x, y)
