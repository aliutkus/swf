# imports
import os
import IDT
from qsketch.sketch import add_sketch_arguments, add_data_arguments
import numpy as np
from qsketch import sketch
from IDT import streamIDT, add_IDT_arguments, add_plotting_arguments
import argparse
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from time import strftime, gmtime
import socket


plt.ion()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Performs iterative distribution transfer'
                                     ' with sliced Wasserstein flow.')
    parser = add_data_arguments(parser)
    parser = add_sketch_arguments(parser)
    parser = add_IDT_arguments(parser)
    parser = add_plotting_arguments(parser)

    parser.add_argument("--output",
                        help="If provided, save the generated samples to "
                             "this file path after transportation. Be sure "
                             "to add a `stop` parameter for the transport "
                             "to terminate for this to happen.")
    parser.add_argument("--stop",
                        help="Number of sketches done before stopping. "
                             "A negative value means endless transport.",
                        type=int,
                        default=-1)
    args = parser.parse_args()

    # load the data
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset
    data_loader = sketch.load_data(args.dataset, args.clip, args.root_data_dir,
                                   args.img_size, args.memory_usage)
    data_dim = int(np.prod(data_loader.dataset[0][0].shape))

    # prepare the projectors
    ProjectorsClass = getattr(sketch, args.projectors)
    projectors = ProjectorsClass(np.inf, args.num_thetas, data_dim)

    # prepare the sketch iterator
    sketches = sketch.SketchIterator(data_loader, projectors, args.batchsize,
                                     args.num_quantiles, 0, args.stop,
                                     args.clip)

    # get the initial samples
    if args.initial_samples is None:
        # If no samples are provided, generate random ones
        samples = np.random.randn(args.num_samples, args.input_dim)*1e-10
        # if required, transform the samples to the data_dim
        if args.input_dim != data_dim:
            np.random.seed(0)
            up_sampling = np.random.randn(args.input_dim, data_dim)
            samples = np.dot(samples, up_sampling)
    else:
        # if a samples file is given, load it
        samples = np.load(args.initial_samples)
        if len(samples.shape) != 2 or samples.shape[1] != data_dim:
            raise ValueError('Samples in %s do not have the right shape. '
                             'They should be num_samples x %d for this '
                             'sketch file.' % (args.initial_samples, data_dim))

    if args.log:
        log_writer = SummaryWriter(os.path.join(args.logdir,
                                                strftime('%Y-%m-%d-%h-%s-',
                                                         gmtime())
                                                + socket.gethostname()))
    else:
        log_writer = None

    if args.plot_target is not None:
        # just handle numpy arrays now
        target_samples = sketch.load_data(args.plot_target, None).dataset.data
        data_dim = target_samples.shape[1]

        ntarget = min(10000, target_samples.shape[0])
        target_samples = target_samples[:ntarget]
        axis_lim = [[v.min(), v.max()] for v in target_samples.T]
    else:
        target_samples = None
        axis_lim = None

    def plot_function(samples, index, error):
        return IDT.base_plot_function(samples, index, error, log_writer,
                                      args, axis_lim, target_samples,
                                      args.contour_every)
        
    samples = streamIDT(sketches, samples, args.stepsize, args.reg,
                        plot_function)

    if args.output is not None:
        np.save(args.output, samples)
