import torch
from torch import nn


class LinearWithBackward(nn.Linear):
    def __init__(self, in_features, out_features, seed=0):
        super(LinearWithBackward, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False)
        self.reset(seed)

    def forward(self, input):
        return super(LinearWithBackward, self).forward(
            input.view(input.shape[0], -1))

    def backward(self, grad):
        return torch.mm(grad.view(grad.shape[0], -1), self.weight)

    def reset(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            gen_device = torch.device('cuda')
        else:
            gen_device = torch.device('cpu')
        new_weights = torch.randn(
            self.weight.shape, device=gen_device).to(self.weight.device)
        self.weight = torch.nn.Parameter(
            new_weights/torch.norm(new_weights, dim=1, keepdim=True))
