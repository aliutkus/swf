# imports
import numpy as np
import sketch
from scipy.interpolate import interp1d
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import os
plt.ion()


def SWFiteration(samples, projector, source_qf, target_qf, quantiles,
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

    # transport the marginals
    transported = np.empty(projections.shape)

    for d in range(num_thetas):
        # code to do 1D transportation
        F = interp1d(source_qf[d], quantiles, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        Ginv = interp1d(quantiles, target_qf[d], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
        zd = np.clip(projections[:, d],
                     source_qf[d, 0], source_qf[d, -1])
        zd = F(zd)
        zd = np.clip(zd, 0, 100)
        transported[:, d] = Ginv(zd)

    noise = np.random.randn(*samples.shape)/np.sqrt(d)

    # do the update
    samples += (stepsize *
                np.dot(transported - projections, projector)/num_thetas
                + np.sqrt(stepsize) * reg
                * noise)
    return samples


def SWF(sketches, samples, stepsize, reg, plot_function):
    index = 0
    for target_qf, projector in sketches:
        print('Transporting, sketch %d' % index, sep=' ', end='\r')
        samples = SWFiteration(samples, projector, None,
                               target_qf, sketches.quantiles,
                               stepsize, reg, index)
        if plot_function is not None:
            plot_function(samples, index)
        index += 1


def base_plot_function(samples, target_samples, index, axis_lim,
                       out_dir, contour_every):
    # contour plot
    if contour_every is not None and not index % contour_every:
        def contour_plot(x, y, fignum, colors, title, axis_lim, *kwargs):
            fig = plt.figure(num=fignum, figsize=(5, 5))
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

        if out_dir and not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if index == 0:
            x0 = target_samples[:, 0]
            y0 = target_samples[:, 1]
            fig = contour_plot(x0, y0, 12, 'Greens', 'Target distribution',
                               axis_lim)
            if out_dir:
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                fig.savefig(os.path.join(out_dir, 'target_distribution.pdf'))
        x = samples[:, 0]
        y = samples[:, 1]
        fig = contour_plot(x, y, 10, 'Blues', 'SWF, iteration %d' % index,
                           axis_lim)
        if out_dir is not None:
            fig.savefig(os.path.join(out_dir,
                                     'output_dist_k=%d.pdf' % (index+1)))
        plt.pause(0.05)
        plt.show()

    else:
        # scatter plot
        fig = plt.figure(num=1, figsize=(5, 5))
        plt.clf()
        sns.set_style("whitegrid")

        plt.figure(1, figsize=(8, 8))
        g = sns.regplot(x=target_samples[:, 0], y=target_samples[:, 1],
                        fit_reg=False, scatter_kws={"color": "black",
                                                    "alpha": 0.3, "s": 2})
        g = sns.regplot(x=samples[:, 0], y=samples[:, 1],
                        fit_reg=False, scatter_kws={"color": "darkred",
                                                    "alpha": 0.3, "s": 2})
        axis_lim = [[l*1.2 for l in v] for v in axis_lim]
        plt.xlim(axis_lim[0])
        plt.ylim(axis_lim[1])
        ticks = [np.linspace(*v, 5) for v in axis_lim]
        g.set(xticks=ticks[0], yticks=ticks[1], yticklabels=[],
              xticklabels=[])
        plt.title('SWF, iteration %d' % index)
        plt.tight_layout()
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            fig.savefig(os.path.join(out_dir,
                        'scatter_output_particles_k=%d.pdf' % (index+1)))
        plt.pause(0.05)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Sliced Wasserstein Flow.')
    parser.add_argument("dataset",
                        help="A file saved by numpy.save "
                             "containing a ndarray of shape "
                             "num_samples x data_dim")
    parser.add_argument("--num_thetas",
                        help="Number of thetas per sketch.",
                        type=int,
                        default=50)
    parser.add_argument("--num_sketches",
                        help="Number of sketches.",
                        type=int,
                        default=400)
    parser.add_argument("--num_quantiles",
                        help="Number of quantiles to compute",
                        type=int,
                        default=100)
    parser.add_argument("--clip",
                        help="Number of datapoints used per sketch. If "
                             "negative, take all of them.",
                        type=int,
                        default=-1)
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
    parser.add_argument("--input_dim",
                        help="Dimension of the random input",
                        type=int,
                        default=100)
    parser.add_argument("--num_samples",
                        help="Number of samples to draw and to transport",
                        type=int,
                        default=3000)
    parser.add_argument("--reg",
                        help="Regularization term",
                        type=float,
                        default=1.)
    parser.add_argument("--stepsize",
                        help="Stepsize",
                        type=float,
                        default=1.)
    parser.add_argument("--plot_dir",
                        help="If given efines an output directory "
                             "for saving the plots.")
    parser.add_argument("--contour_every",
                        help="Will regularly plot a contour plot, if provided",
                        type=int)


    args = parser.parse_args()

    # load the data
    data = np.load(args.dataset)
    data_dim = data.shape[-1]

    # prepare the projectors
    projectors = sketch.Projectors(args.num_thetas, data_dim)

    # prepare the sketch iterator
    sketches = sketch.SketchIterator(data, projectors,
                                     args.num_quantiles, args.stop,
                                     args.clip)

    # If no samples are provided, generate random ones
    samples = np.random.randn(args.num_samples, args.input_dim)*1e-10
    # if required, transform the samples to the data_dim
    if args.input_dim != data_dim:
        np.random.seed(0)
        up_sampling = np.random.randn(args.input_dim, data_dim)
        samples = np.dot(samples, up_sampling)

    ntarget = min(10000, data.shape[0])
    target_samples = data[:ntarget]
    axis_lim = [[v.min(), v.max()] for v in target_samples.T]

    def plot_function(samples, index):
        return base_plot_function(samples, target_samples, index,
                                  axis_lim, args.plot_dir, args.contour_every)

    samples = SWF(sketches, samples, args.stepsize, args.reg,
                  plot_function)

    if args.output is not None:
        np.save(args.output, samples)
