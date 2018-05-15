# imports
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import copy
from qsketch import sketch
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from time import strftime, gmtime
import socket
import os



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
                 stepsize, reg, writer, index):
    projector = np.reshape(projector, [-1, samples.shape[1]])
    num_thetas = projector.shape[0]

    # compute the projections
    projections = np.dot(samples, projector.T)

    if source_qf is None:
        # compute the quantile function in the projected domain
        source_qf = sketch.fast_percentile(projections.T, quantiles)

    source_qf = np.reshape(source_qf, [-1, len(quantiles)])
    target_qf = np.reshape(target_qf, [-1, len(quantiles)])

    if writer is not None:
        writer.add_scalar('loss', np.mean((source_qf-target_qf)**2), index)
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

    shape_noise = samples.shape
    noise = np.random.randn(*shape_noise)
    # from scipy.stats import levy_stable
    # noise = levy_stable.rvs(alpha=1.5, beta=0, size= shape_noise)

    samples += (stepsize *
                np.dot(transported - projections, projector)/num_thetas
                + np.sqrt(stepsize) * reg
                * noise)

    return samples, source_qf


def batchIDT(target_qf, projectors, num_quantiles, chain_in,
             samples, plot_function, logdir, compute_chain_out=True):

    # prepare the logger
    if logdir is not None:
        log_writer = SummaryWriter(os.path.join(logdir,
                                                strftime('%Y-%m-%d-%h-%s-',
                                                         gmtime())
                                                + socket.gethostname()))
    else:
        log_writer = None

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

    current = 0
    for epoch in range(chain_in.epochs):
        # for each epoch, loop over the sketches
        for sketch_indexes in tqdm.tqdm(projectors_loader,
                                        desc='epoch %d'%epoch):
            if chain_in.qf is None:
                source_qf = None
            else:
                source_qf = chain_in.qf[epoch, sketch_indexes]

            (samples,
             source_qf) = IDTiteration(samples, projectors[sketch_indexes],
                                       source_qf, target_qf[sketch_indexes],
                                       quantiles, chain_in.stepsize,
                                       chain_in.reg, log_writer, current)

            if compute_chain_out:
                source_qf = np.reshape(source_qf,
                                       [len(sketch_indexes),
                                        data_dim, num_quantiles])
                chain_out.qf[epoch, sketch_indexes] = source_qf

            current += 1
            if plot_function is not None:
                plot_function(samples, current)

    if not compute_chain_out:
        chain_out = None
    return samples, chain_out


def streamIDT(sketches, samples, stepsize, reg, plot_function, logdir):
    # prepare the logger
    if logdir is not None:
        log_writer = SummaryWriter(os.path.join(logdir,
                                                strftime('%Y-%m-%d-%h-%s-',
                                                         gmtime())
                                                + socket.gethostname()))
    else:
        log_writer = None
    index = 0
    for target_qf, projector in sketches:
        index += 1
        print('Transporting, sketch %d' % index, sep=' ', end='\r')
        samples = IDTiteration(samples, projector, None, target_qf,
                               sketches.quantiles, stepsize, reg, log_writer,
                               index)[0]
        if plot_function is not None:
            #import ipdb; ipdb.set_trace()
            plot_function(samples, index)


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
    parser.add_argument("--logdir",
                        help="Directory for the logs using tensorboard. If "
                             "not provided, will not use this feature.")
    return parser
