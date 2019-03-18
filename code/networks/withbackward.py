import torch
from torch import nn


class LinearWithBackward(nn.Linear):
    def __init__(self, in_features, out_features):
        super(LinearWithBackward, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False)
        self.weight = torch.nn.Parameter(
            self.weight/torch.norm(self.weight, dim=1, keepdim=True))

    def forward(self, input):
        return super(LinearWithBackward, self).forward(
            input.view(input.shape[0], -1))

    def backward(self, grad):
        return torch.mm(grad.view(grad.shape[0], -1), self.weight)
