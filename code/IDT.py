# imports
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import copy
from qsketch import sketch
from scipy.interpolate import interp1d


class Chain:

    def __init__(self, batchsize, epochs, stepsize=1, reg=1):
        self.batchsize = batchsize
        self.epochs = epochs
        self.stepsize = stepsize
        self.reg = reg
        self.qf = None

    def __copy__(self):
        return Chain(self.batchsize, self.epochs, self.stepsize, self.reg)


def IDTiteration(samples, projector, source_qf, target_qf, quantiles,
                 stepsize, reg):
    projector = np.reshape(projector, [-1, samples.shape[1]])
    num_thetas = projector.shape[0]

    # compute the projections
    projections = np.dot(samples, projector.T)

    if source_qf is None:
        # compute the quantile function in the projected domain
        source_qf = sketch.fast_percentile(projections.T, quantiles)

    source_qf = np.reshape(source_qf, [-1, len(quantiles)])
    target_qf = np.reshape(target_qf, [-1, len(quantiles)])

    # transport the marginals
    transported = np.empty(projections.shape)

    for d in range(num_thetas):
        F = interp1d(source_qf[d], quantiles, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        Ginv = interp1d(quantiles, target_qf[d], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
        zd = np.clip(projections[:, d],
                     source_qf[d, 0], source_qf[d, -1])
        zd = F(zd)
        zd = np.clip(zd, 0, 100)
        transported[:, d] = Ginv(zd)

    samples += (stepsize *
                np.dot(transported - projections, projector)/num_thetas
                + np.sqrt(stepsize) * reg
                * np.random.randn(1, samples.shape[1]))
                #* np.random.randn(*samples.shape))

    return samples, source_qf


def batchIDT(target_qf, projectors, num_quantiles, chain_in,
             samples, plot_function, compute_chain_out=True):

    # prepare the projectors
    projectors_loader = DataLoader(range(len(projectors)),
                                   batch_size=chain_in.batchsize,
                                   shuffle=True)
    quantiles = np.linspace(0, 100, num_quantiles)
    data_dim = projectors.data_dim

    if compute_chain_out:
        # prepare the chain_out
        chain_out = copy.copy(chain_in)
        chain_out.qf = np.empty((chain_in.epochs,
                                 chain_in.num_sketches,
                                 data_dim, num_quantiles))

    for epoch in tqdm.tqdm(range(chain_in.epochs)):
        # for each epoch, loop over the sketches
        for index, sketch_indexes in enumerate(projectors_loader):
            if chain_in.qf is None:
                source_qf = None
            else:
                source_qf = chain_in.qf[epoch, sketch_indexes]

            (samples,
             source_qf) = IDTiteration(samples, projectors[sketch_indexes],
                                       source_qf, target_qf[sketch_indexes],
                                       quantiles, chain_in.stepsize,
                                       chain_in.reg)

            if compute_chain_out:
                source_qf = np.reshape(source_qf,
                                       [len(sketch_indexes),
                                        data_dim, num_quantiles])
                chain_out.qf[epoch, sketch_indexes] = source_qf

            if plot_function is not None:
                plot_function(samples, epoch, index)

    if not compute_chain_out:
        chain_out = None
    return samples, chain_out


def streamIDT(sketches, samples, stepsize, reg, plot_function):
    index = 0
    for target_qf, projector in sketches:
        print('Transporting, sketch %d' % index, sep=' ', end='\r')
        samples = IDTiteration(samples, projector, None, target_qf,
                               sketches.quantiles, stepsize, reg)[0]
        if plot_function is not None:
            plot_function(samples, 0, index)
        index += 1



def add_IDT_arguments(parser):
    parser.add_argument("--initial_samples",
                        help="Initial samples to use, must be a file"
                             "containing a ndarray of dimension num_samples x "
                             "dim, saved with numpy. If provided, overrides"
                             "the parameters `input_dim`, `num_samples`")
    parser.add_argument("--input_dim",
                        help="Dimension of the random input",
                        type=int,
                        default=100)
    parser.add_argument("--num_samples",
                        help="Number of samples to draw and to transport",
                        type=int,
                        default=3000)
    parser.add_argument("--batchsize",
                        help="Number of sketches to use per step.",
                        type=int,
                        default=1)
    parser.add_argument("--reg",
                        help="Regularization term",
                        type=float,
                        default=1.)
    parser.add_argument("--stepsize",
                        help="Stepsize",
                        type=float,
                        default=1.)
    return parser
