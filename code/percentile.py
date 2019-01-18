import torch


class Percentile(torch.autograd.Function):
    """
    torch Percentile autograd Functions subclassing torch.autograd.Function
    """

    def __init__(self, nb_percentiles, device='cpu'):
        self.nb_percentiles = nb_percentiles
        self.percentiles = torch.linspace(0, 100, nb_percentiles).to(device)
        self.device = device

    def forward(self, input):
        """
        Find the percentile of a list of values.
        """
        in_sorted, in_argsort = torch.sort(input, dim=1)
        positions = self.percentiles * (input.shape[1]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[1] - 1] = input.shape[1] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        # identical = (ceiled == floored)
        # weight_floored[identical] = 0.5
        # weight_ceiled[identical] = 0.5
        d0 = in_sorted[:, floored.long()] * weight_floored
        d1 = in_sorted[:, ceiled.long()] * weight_ceiled
        self.save_for_backward(in_argsort, floored.long(), ceiled.long(),
                               weight_floored, weight_ceiled)
        return d0+d1

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors
        input_shape = in_argsort.shape

        # the argsort in the flattened in vector
        rows_offsets = (
            input_shape[1]
            * torch.range(
                    0, input_shape[0]-1, device=in_argsort.device)
            )[:, None].long()
        in_argsort = (in_argsort + rows_offsets).view(-1)
        floored = (floored + rows_offsets).view(-1).long()
        ceiled = (ceiled + rows_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[None, :]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[None, :]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input
