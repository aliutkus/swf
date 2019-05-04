import torch
import torch.nn as nn
from torchpercentile import Percentile


def randn(shape):
    if torch.cuda.is_available():
        gen_device = torch.device('cuda')
    else:
        gen_device = torch.device('cpu')
    result = torch.randn(
        shape, device=gen_device)
    return result


class LinearProjector(nn.Linear):
    def __init__(self, shape_in, num_out):
        super(LinearProjector, self).__init__(
            in_features=torch.prod(torch.tensor(shape_in)),
            out_features=num_out,
            bias=False)
        self.reset_parameters()

    def forward(self, input):
        return super(LinearProjector, self).forward(
            input.view(input.shape[0], -1))

    def reset_parameters(self):
        super(LinearProjector, self).reset_parameters()
        self.weight = torch.nn.Parameter(
            self.weight/torch.norm(self.weight, dim=1, keepdim=True))

    def backward(self, grad):
        """Manually compute the gradient of the layer for any input"""
        return torch.mm(grad.view(grad.shape[0], -1), self.weight)


class SparseLinearProjector(LinearProjector):
    """ A linear projector such that only a few proportion of its elements
    are active"""

    def __init__(self, shape_in, num_out, density=5):
        self.density = density
        super(SparseLinearProjector, self).__init__(
            shape_in=shape_in,
            num_out=num_out)
        self.reset_parameters()

    def reset_parameters(self):
        super(SparseLinearProjector, self).reset_parameters()
        threshold = Percentile()(self.weight.abs().flatten()[:, None],
                                 [100-self.density, ])
        new_weights = self.weight.clone()
        new_weights[torch.abs(self.weight) < threshold] = 0
        self.weight = torch.nn.Parameter(
            new_weights/torch.norm(new_weights, dim=1, keepdim=True))
        return self


import numpy as np
from functools import reduce

class LocalProjector(LinearProjector):
    """Each projector is a set of unit-length random vectors, that are
    active only in neighbourhoods of the data"""
    @staticmethod
    def get_factors(n):
        return np.unique(reduce(list.__add__,
                         ([i, int(n//i)] for i in range(1, int(n**0.5) + 1)
                          if not n % i)))[1:]

    def __init__(self, shape_in, num_out, index=0):
        self.data_dim = torch.prod(torch.tensor(shape_in)).item()
        self.factors = LocalProjector.get_factors(self.data_dim)
        super(LocalProjector, self).__init__(
            shape_in=shape_in,
            num_out=num_out,
            index=index)
        self.recycle(index)

    def recycle(self, index):
        # generate each set of projectors
        (out_dim, in_dim) = self.weight.shape
        new_weight = torch.zeros_like(self.weight)
        torch.manual_seed(index)
        size_patches = self.factors[
                        torch.randint(high=len(self.factors), size=(1,)).item()]
        short_matrix = randn((out_dim, size_patches)).to(new_weight.device)
        short_matrix = short_matrix/torch.norm(
                        short_matrix, dim=1, keepdim=True)
        pos = 0
        pos_col = 0
        while pos < out_dim:
            row_indices = slice(pos, pos + size_patches)
            col_indices = slice(pos_col, pos_col + size_patches)
            pos += size_patches
            pos_col += size_patches
            if pos_col >= in_dim:
                pos_col = 0
            new_weight[row_indices, col_indices] = short_matrix[row_indices, :]
        self.weight = torch.nn.Parameter(new_weight)
