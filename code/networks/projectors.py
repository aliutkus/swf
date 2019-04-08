import torch
import torch.nn as nn
from torchpercentile import Percentile


class LinearProjector(nn.Linear):
    def __init__(self, shape_in, num_out, index=0):
        super(LinearProjector, self).__init__(
            in_features=torch.prod(torch.tensor(shape_in)),
            out_features=num_out,
            bias=False)
        self.recycle(index)

    def forward(self, input):
        return super(LinearProjector, self).forward(
            input.view(input.shape[0], -1))

    def recycle(self, index):
        """ resetting the weights of the module in a reproductible way.
        The difficulty lies in the fact that the random sequences on
        CPU and GPU are not the same, even with the same seed."""
        torch.manual_seed(index)
        if torch.cuda.is_available():
            gen_device = torch.device('cuda')
        else:
            gen_device = torch.device('cpu')
        new_weights = torch.randn(
            self.weight.shape, device=gen_device).to(self.weight.device)
        self.weight = torch.nn.Parameter(
            new_weights/torch.norm(new_weights, dim=1, keepdim=True))

    def backward(self, grad):
        """Manually compute the gradient of the layer for any input"""
        return torch.mm(grad.view(grad.shape[0], -1), self.weight)


class SparseLinearProjector(LinearProjector):
    """ A linear projector such that only a few proportion of its elements
    are active"""
    def __init__(self, shape_in, num_out, density=5, index=0):
        self.density = density
        super(SparseLinearProjector, self).__init__(
            shape_in=shape_in,
            num_out=num_out,
            index=index)
        self.recycle(index)

    def recycle(self, index):
        super(SparseLinearProjector, self).recycle(index)
        threshold = Percentile()(self.weight.abs().flatten()[:, None],
                                 [100-self.density, ])
        new_weights = self.weight.clone()
        new_weights[torch.abs(self.weight) < threshold] = 0
        self.weight = torch.nn.Parameter(
            new_weights/torch.norm(new_weights, dim=1, keepdim=True))
        return self
