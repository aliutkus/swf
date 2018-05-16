# imports
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import copy
from qsketch import sketch
from scipy.interpolate import interp1d
from torchvision.utils import save_image, make_grid
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import os
plt.ion()


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
                 stepsize, reg, index):
    projector = np.reshape(projector, [-1, samples.shape[1]])
    num_thetas = projector.shape[0]

    # compute the projections
    projections = np.dot(samples, projector.T)

    if source_qf is None:
        # compute the quantile function in the projected domain
        source_qf = sketch.fast_percentile(projections.T, quantiles)

    source_qf = np.reshape(source_qf, [-1, len(quantiles)])
    target_qf = np.reshape(target_qf, [-1, len(quantiles)])

    # sliced wasserstein distance
    sw_error = np.mean((source_qf-target_qf)**2)

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
    noise = np.random.randn(*shape_noise)/np.sqrt(d)
    # from scipy.stats import levy_stable
    # noise = levy_stable.rvs(alpha=1.5, beta=0, size= shape_noise)

    samples += (stepsize *
                np.dot(transported - projections, projector)/num_thetas
                + np.sqrt(stepsize) * reg
                * noise)

    return samples, source_qf, sw_error


def batchIDT(target_qf, projectors, num_quantiles, chain_in,
             samples, plot_function, compute_chain_out=True):

    # prepare the projectors
    import torch
    torch.manual_seed(7)
    projectors_loader = DataLoader(range(len(projectors)),
                                   batch_size=chain_in.batchsize,
                                   shuffle=True)
    quantiles = np.linspace(0, 100, num_quantiles)
    [num_sketches, num_thetas, num_quantiles] = target_qf.shape
    if compute_chain_out:
        # prepare the chain_out
        chain_out = copy.copy(chain_in)
        chain_out.qf = np.empty((chain_in.epochs,
                                 num_sketches, num_thetas, num_quantiles))

    current = 0
    for epoch in range(chain_in.epochs):
        # for each epoch, loop over the sketches
        for sketch_indexes in tqdm.tqdm(projectors_loader,
                                        desc='epoch %d' % epoch):
            if chain_in.qf is None:
                source_qf = None
            else:
                source_qf = chain_in.qf[epoch, sketch_indexes]

            (samples,
             source_qf,
             sw_error) = IDTiteration(samples, projectors[sketch_indexes],
                                      source_qf, target_qf[sketch_indexes],
                                      quantiles, chain_in.stepsize,
                                      chain_in.reg, current)
            if compute_chain_out:
                source_qf = np.reshape(source_qf,
                                       [len(sketch_indexes),
                                        num_thetas, num_quantiles])
                chain_out.qf[epoch, sketch_indexes] = source_qf

            if plot_function is not None:
                plot_function(samples, current, sw_error)
            current += 1

    if not compute_chain_out:
        chain_out = None
    return samples, chain_out


def streamIDT(sketches, samples, stepsize, reg, plot_function):
    # prepare the logger
    index = 0
    for target_qf, projector in sketches:
        print('Transporting, sketch %d' % index, sep=' ', end='\r')
        (samples, _, sw_error) = IDTiteration(samples, projector, None,
                                              target_qf, sketches.quantiles,
                                              stepsize, reg, index)
        if plot_function is not None:
            plot_function(samples, index, sw_error)
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


def add_plotting_arguments(parser):
    parser.add_argument("--plot",
                        help="Flag indicating whether or not to plot samples",
                        action="store_true")
    parser.add_argument("--log",
                        help="Flag indicating whether or not to log the "
                             "sliced Wasserstein error along iterations.",
                        action="store_true")
    parser.add_argument("--plot_target",
                        help="Samples from the target. Same constraints as "
                             "the `dataset` argument.")
    parser.add_argument("--logdir",
                        help="Directory for the logs using tensorboard. If "
                             "not provided, will log to directory `logs`.",
                        default="logs")
    parser.add_argument("--plot_dir",
                        help="Output directory for saving the plots")
    return parser


def base_plot_function(samples, index, error, log_writer, args, axis_lim,
                       target_samples):
    
    # hacky code to plot the SW cost
    """if os.path.exists('logs.npy'):
        data = np.load('logs.npy').item()
        if index < data['errors'][-1].size:
            data['errors'] += [np.array([error])]
        else:
            data['errors'][-1] = np.concatenate((data['errors'][-1], [error]))
    else:
        data = {'errors': [np.array([error])]}
    np.save('logs.npy', data)"""

    if log_writer is not None:
        log_writer.add_scalar('data/loss', error, index)

    if not args.plot and not args.plot_dir:
        return

    data_dim = samples.shape[-1]
    image = False

    # try to identify if it's an image or not
    if data_dim > 700:
        # if the data dimension is large: probably an image.
        square_dim_bw = np.sqrt(data_dim)
        square_dim_col = np.sqrt(data_dim/3)
        if not (square_dim_col % 1):  # check monochrome
            image = True
            nchan = 3
            img_dim = int(square_dim_col)
        elif not (square_dim_bw % 1):  # check color
            image = True
            nchan = 1
            img_dim = int(square_dim_bw)

    if not image:
        contour = True
        if contour:
            def contour_plot(x, y, fignum, colors, title, axis_lim, *kwargs):
                fig = plt.figure(num=fignum, figsize=(2, 2))
                plt.clf()

                # Basic 2D density plot
                sns.set_style("whitegrid")

                g = sns.kdeplot(x, y, cmap=colors, shade=True, shade_lowest=False,
                                *kwargs)
                axis_lim = [[l*1.2 for l in v] for v in axis_lim]
                plt.xlim(axis_lim[0])
                plt.ylim(axis_lim[1])
                ticks = [np.linspace(*v, 5) for v in axis_lim]
                g.set(xticks=ticks[0], yticks=ticks[1], yticklabels=[],
                      xticklabels=[])
                if title is not None:
                    plt.title(title)
                plt.tight_layout()
                return fig

            if args.plot_dir and not os.path.exists(args.plot_dir):
                os.mkdir(args.plot_dir)

            if index == 0:
                x0 = target_samples[:, 0]
                y0 = target_samples[:, 1]
                fig = contour_plot(x0, y0, 2, 'Greens', None, axis_lim)
                if args.plot_dir:
                    fig.savefig(os.path.join(args.plot_dir,
                                             'target.pdf'))
            x = samples[:, 0]
            y = samples[:, 1]
            fig = contour_plot(x, y, 1, 'Blues', None, axis_lim)
            if args.plot:
                plt.pause(0.05)
                plt.show()
            if args.plot_dir:
                fig.savefig(os.path.join(args.plot_dir,
                                         'output_dist_k=%d.pdf' % (index+1)))

        else:
            # no image: just plot second data dimension vs first one
            fig = plt.figure(num=1, figsize=(2, 2))
            plt.clf()

            # Basic 2D density plot
            sns.set_style("whitegrid")

            plt.figure(1, figsize=(8, 8))
            if args.plot_target is not None:
                g = sns.regplot(x=target_samples[:, 0], y=target_samples[:, 1],
                                fit_reg=False, scatter_kws={"color":"black",
                                                            "alpha":0.3,"s":2} )
            g = sns.regplot(x=samples[:, 0], y=samples[:, 1],
                            fit_reg=False, scatter_kws={"color":"darkred",
                                                        "alpha":0.3,"s":2} )
            #g = plt.plot(samples[:, 0], samples[:, 1], 'ob')
            axis_lim = [[l*1.2 for l in v] for v in axis_lim]
            plt.xlim(axis_lim[0])
            plt.ylim(axis_lim[1])
            ticks = [np.linspace(*v, 5) for v in axis_lim]
            g.set(xticks=ticks[0], yticks=ticks[1], yticklabels=[],
                  xticklabels=[])
            plt.tight_layout()
            #plt.title('Sketch %d'
            #          % (index+1))
            if args.plot:
                plt.pause(0.05)
                plt.show()
            if args.plot_dir:
                fig.savefig(os.path.join(args.plot_dir,
                            'scatter_output_particles_k=%d.pdf' % (index+1)))
        return

    # it's an image, output a grid of samples and writes it to disk
    if index % 50:
        return

    [num_samples, data_dim] = samples.shape
    samples = samples[:min(208, num_samples)]
    num_samples = samples.shape[0]

    samples = np.reshape(samples,
                         [num_samples, nchan, img_dim, img_dim])
    pic = make_grid(torch.Tensor(samples),
                    nrow=8, padding=2, normalize=True, scale_each=True)
    if not os.path.exists(args.plot_dir):
        os.mkdir(args.plot_dir)
    save_image(pic, '{}/image_{}.png'.format(args.plot_dir, index))
