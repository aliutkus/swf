# all the dirty plotting stuff goes here

import seaborn as sb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import numpy as np
from math import floor
import itertools
import os
sb.set_style(sb.axes_style('whitegrid'))


def find_closest(items, dataset):
    """ find the indices of the entries of the dataset that are closest
    to the provided items"""
    # bring the items to cpu and flatten them
    items = items.detach().cpu()
    num = items.shape[0]
    items = items.view(num, -1)

    # initialize the losses to infinite
    mindist = torch.ones(num)*float('inf')
    closest = torch.zeros_like(items)
    num_workers = max(1, floor((mp.cpu_count()-2)/2))
    dataloader = DataLoader(dataset, batch_size=5000, num_workers=num_workers)
    for (candidates, _) in dataloader:
        candidates = candidates.view(candidates.shape[0], -1).cpu()
        distances = torch.norm(
                        items[:, None, :] - candidates[None, ...],
                        dim=-1)
        mindist_in_batch, closest_in_batch = torch.min(distances, dim=1)
        replace = torch.nonzero(mindist_in_batch < mindist)
        closest[replace] = candidates[closest_in_batch[replace]]
    return closest


def plot_function(data, axes, markers='r.'):
    """ Plot some data to some axes. Checks whether it's image or
    scatter plot"""
    data = data.detach().cpu()
    if len(data.shape) == 4:
        # it's image data
        pic = make_grid(
            data,
            nrow=16, padding=2, normalize=True, scale_each=True
            )
        pic_npy = pic.numpy()
        newplots = [axes.imshow(
            np.transpose(pic_npy, (1, 2, 0)),
            interpolation='nearest')]
    elif len(data.squeeze().shape) == 2 and data.squeeze().shape[1] == 2:
        # it's 2D data
        newplots = plt.plot(data.squeeze().numpy()[:, 0],
                            data.squeeze().numpy()[:, 1],
                            markers, markersize=1)
    return newplots


class SWFPlot:
    """ some big dirty class with just plotting stuff"""
    def __init__(self, ndim, train_data, plot_dir,
                 plot_every, match_every,
                 decode_fn, nb_plot, nb_match, test):
        """
        Initialize the plotting class

        ndim: int,
            number of dimensions to consider for the density plots
            and the projections
        train_data: dataset
            dataset to use for finding the closest entries, and also for the
            density plots
        plot_dir: string
            directory where to put the generated images to
        plot_every: int
            will plot each time the epoch is a multiple of this
        match_every: int_train
            will look for the closest entries each time the epoch is a
            multiple of this
        decode_fn: function or None
            if not None, the features are sent there for plotting
        nb_plot: int
            number of samples to plot (use -1 for all)
        nb_match: int
            number of samples of which to find closest in dataset
        test: """

        # hard coded switches
        density_plot = True
        particles_plot = True
        closest_plot = True
        projections_plot = True

        self.plot_every = plot_every
        self.match_every = match_every
        self.ndim = ndim
        self.train_data = train_data
        self.plot_dir = plot_dir
        self.decode_fn = decode_fn
        self.nb_match = nb_match
        self.nb_plot = nb_plot
        self.plots_to_purge = []

        self.axes = {}

        # initialize the figures if needed
        if density_plot:
            # Plot figure with subplots of different sizes
            fig = plt.figure(1)
            fig.clf()

            # prepare the heat map of the distribution of the data
            train_loader = DataLoader(train_data, batch_size=len(train_data))
            data = next(iter(train_loader))[0].numpy()

            self.axes['density'] = []
            for row in range(ndim - 1):
                row_axes = []
                for col in range(row+1):
                    ax = fig.subplot(ndim, ndim, row*ndim+col)
                    ax.autoscale(True, tight=True)
                    ax.kdeplot(
                        data[:, row],
                        data[:, col],
                        gridsize=100, n_levels=200,
                        linewidths=0.1)
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                    row_axes += [ax]
                self.axes['density'] += [row_axes]
            self.figs['density'] = fig
            fig.savefig('density_fig_init.pdf')

        def new_fig(num):
            fig = plt.figure(num)
            fig.autoscale(True, tight=True)
            fig.clf()
            axes = plt.gca()
            axes.autoscale(True, tight=True)
            axes.xaxis.set_major_formatter(plt.NullFormatter())
            axes.yaxis.set_major_formatter(plt.NullFormatter())
            return (fig, axes)
        if particles_plot:
            (self.figs['particles'], self.axes['particles']) = new_fig(2)
        if closest_plot:
            (self.figs['closest'], self.axes['closest']) = new_fig(3)
        if projections_plot:
            (self.figs['projections'], self.axes['projections']) = new_fig(4)

    def save_figs(self, filename):
        # create the folder if it doesn't exist
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        # now loop over the available figures and plot them all
        for name in self.figs:
            self.figs[name].canvas.draw()
            self.figs[name].savefig(os.path.join(self.plot_dir, name+filename))

    def clear_plots(self):
        for plot in self.new_plots:
            plot.remove()
        self.new_plots = []

    def logger(self, vars, epoch):
        # checking whether we need to match and/or plot
        match = (self.match_every > 0
                 and epoch > 0 and not epoch % self.match_every)
        plot = (epoch < 0
                or (self.plot_every > 0 and not epoch % self.plot_every))

        if not plot and not match:
            # nothing to do
            return

        # convert the particles to numpy
        train = vars['particles']['train'].squeeze().numpy()
        test = (vars['particles']['test'].squeeze().numpy()
                if 'test' in vars['particles']
                else None)

        self.clear_plots()
        newplots = []
        if 'density' in self.figs:
            # plot the actual values of the particles over density plots,
            # just doing this for train particles
            for dim2 in range(1, self.ndim):
                for dim1 in range(dim2):
                    ax = self.axes['density'][dim1-1][dim2]
                    newplots += ax.plot(train[:, dim1], train[:, dim2],
                                        'xb', markersize=1)


        # prepares the generated outputs and their closest match in dataset
        # if required
        for task in vars['particles']:
            if match:
                # create empty grid
                writer.write("Finding closest matches in dataset")
                closest = find_closest(vars['particles'][task][:nb_match],
                                       dataset)
            else:
                closest = None

            # if we use the autoencoder deocde the particles to visualize them
            plot_data = vars['particles'][task][:nb_plot]

            if decode_fn is not None:
                plot_data = decode_fn(plot_data.to('cpu'))
                plot_function(data=plot_data,
                              axes=axes,
                              plot_dir=plot_dir,
                              filename='%s_%04d.png' % (task, index))

            if closest is not None:
                closest = decode_fn(closest)
                plot_function(data=closest,
                              axes=axes,
                              plot_dir=plot_dir,
                              filename='%s_match_%04d.png' % (task, index))
