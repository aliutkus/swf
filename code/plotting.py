# all the dirty plotting stuff goes here

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import numpy as np
from math import floor
import tqdm
import os
#sb.set_style(sb.axes_style('whitegrid'))


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
        plt.axis('off')
    elif len(data.squeeze().shape) == 2 and data.squeeze().shape[1] == 2:
        # it's 2D data
        newplots = plt.plot(data.squeeze().numpy()[:, 0],
                            data.squeeze().numpy()[:, 1],
                            markers, markersize=1)
    plt.subplots_adjust()
    return newplots


class SWFPlot:
    """ some big dirty class with just plotting stuff"""
    def __init__(self, features, dataset, plot_dir,
                 plot_every, match_every,
                 decode_fn, nb_plot, test):
        """
        Initialize the plotting class

        ndim: int,
            number of dimensions to consider for the density plots
        dataset: dataset
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
        test: """

        # hard coded switches
        self.density_plot = True
        self.particles_plot = True
        self.closest_plot = True
        self.projections_plot = True

        self.plot_every = plot_every
        self.match_every = match_every
        self.ndim = features if isinstance(features, int) else len(features)
        self.features = features
        self.dataset = dataset
        self.plot_dir = plot_dir
        self.decode_fn = decode_fn
        self.nb_plot = nb_plot

        self.plots_to_purge = []

        self.axes = {}
        self.figs = {}
        self.updated = []
        # initialize the figures if needed
        if self.density_plot:
            self.density_nlevels = 10
            self.density_data_palette = get_cmap("Blues")
            self.density_train_palette = get_cmap("copper")

            # Plot figure with subplots of different sizes
            fig = plt.figure(1, dpi=200)
            fig.clf()

            # prepare the heat map of the distribution of the data
            train_loader = DataLoader(dataset, batch_size=min(50000,len(dataset)))
            data = next(iter(train_loader))[0].squeeze().numpy()

            std_data = np.std(data, axis=0)
            data += (np.random.randn(*data.shape)
                     * np.maximum(1e-5,
                                  std_data[None, :]/100))
            if isinstance(features, int) and features == data.shape[1]:
                self.features = range(self.features)
            elif isinstance(features, int) and features > data.shape[1]:
                raise Exception('There are less features than those asked for')
            elif isinstance(features, int) and features <= data.shape[1]:
                # picking the most energetic features
                self.features = np.argsort(std_data)[-self.ndim:][::-1]

            self.axes['density'] = []
            print('Preparing the density plots of figures... may take a while')
            pbar = tqdm.tqdm(total=(self.ndim - 1)*self.ndim/2)
            for row in range(self.ndim - 1):
                row_axes = []
                for col in range(row+1):
                    xlims = np.percentile(data[:, self.features[col]],
                                          [1, 80])
                    ylims = np.percentile(data[:, self.features[row+1]],
                                          [1, 80])
                    ax = plt.subplot(self.ndim-1, self.ndim-1,
                                     row*(self.ndim-1)+col+1)
                    sb.kdeplot(
                        data=data[:, self.features[col]],
                        data2=data[:, self.features[row+1]],
                        gridsize=100, n_levels=self.density_nlevels,
                        cut=0.7,
                        shade=True,
                        shade_lowest=False,
                        cmap=self.density_data_palette,
                        ax=ax)
                    pbar.update(1)
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                    ax.yaxis.set_major_formatter(plt.NullFormatter())
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
                    row_axes += [ax]
                    if row == self.ndim - 2:
                        plt.text(0.5, -(self.ndim-1)/15.,
                                 'feature %d' % self.features[col],
                                 transform=ax.transAxes,
                                 horizontalalignment='center')

                plt.text(1+(self.ndim-1)/30, 0.5,
                         'feature %d' % self.features[row+1], rotation=90,
                         transform=ax.transAxes,
                         verticalalignment='center')
                self.axes['density'] += [row_axes]

            self.density_legend = [
                mpatches.Patch(
                    color=self.density_data_palette(255),
                    label='Training data'),
                mpatches.Patch(
                    color=self.density_train_palette(255),
                    label='Generated particles')]
            fig.suptitle('Density plot of particles')
            plt.figlegend(handles=self.density_legend, loc='upper center',
                          frameon=True, fancybox=True, ncol=2, bbox_to_anchor=(0.4, 0.92))
            plt.subplots_adjust()
            #plt.tight_layout()
            self.figs['density'] = fig
            fig.canvas.draw()
            self.updated += ['density']
            self.save_figs('init.pdf')

        def new_fig(num):
            fig = plt.figure(num, dpi=200)
            fig.clf()
            axes = plt.gca()
            axes.autoscale(True, tight=True)
            axes.xaxis.set_major_formatter(plt.NullFormatter())
            axes.yaxis.set_major_formatter(plt.NullFormatter())
            plt.subplots_adjust()
            return (fig, axes)
        if self.particles_plot:
            (self.figs['particles_train'],
                self.axes['particles_train']) = new_fig(2)
            (self.figs['particles_test'],
                self.axes['particles_test']) = new_fig(3)
        if self.closest_plot:
            (self.figs['closest'], self.axes['closest']) = new_fig(4)
        if self.projections_plot:
            (self.figs['projections'], self.axes['projections']) = new_fig(5)
        self.updated = []

    def save_figs(self, filename):
        # create the folder if it doesn't exist
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        # now loop over the available figures and plot them all
        for name in self.updated:
            self.figs[name].canvas.draw()
            self.figs[name].savefig(
                os.path.join(self.plot_dir, name+filename),
                bbox_inches='tight')
        self.updated = []

    def clear_plots(self):
        for plot in self.plots_to_purge:
            plot.remove()
        self.plots_to_purge = []

    def log(self, vars, epoch):
        # checking whether we need to match and/or plot
        match = (self.match_every > 0
                 and epoch > 0 and not epoch % self.match_every)
        plot = (epoch == 0
                or (self.plot_every > 0 and not epoch % self.plot_every))

        if not plot and not match:
            # nothing to do
            return

        # convert the particles to cpu
        train = vars['particles']['train'].to('cpu')
        test = (vars['particles']['test'].to('cpu')
                if 'test' in vars['particles']
                else None)

        self.clear_plots()
        newplots = []
        self.updated = []
        if 'density' in self.figs:
            # plot the actual values of the particles over density plots,
            # just doing this for train particles
            for row in range(self.ndim - 1):
                for col in range(row+1):
                    ax = self.axes['density'][row][col]
                    xlimits = ax.get_xlim()
                    ylimits = ax.get_ylim()
                    children = ax.get_children()
                    sb.kdeplot(
                        data=train.squeeze()[:, self.features[col]].numpy(),
                        data2=train.squeeze()[:, self.features[row+1]].numpy(),
                        gridsize=100, n_levels=self.density_nlevels,
                        shade=False,
                        cmap=self.density_train_palette,
                        ax=ax)

                    ax.autoscale(enable=True, tight=True)
                    ax.set_xlim(*xlimits)
                    ax.set_ylim(*ylimits)
                    newplots += [p for p in ax.get_children() if p not in children]
            self.figs['density'].suptitle(
                'Density plot of particles, iteration %04d' % epoch)
            self.updated += ['density']
        train = train[:self.nb_plot]
        if test is not None:
            test = test[:self.nb_plot]
        if self.decode_fn is not None:
            train = self.decode_fn(train)
            if test is not None:
                test = self.decode_fn(test)
        if 'particles_train' in self.figs:
            plot_function(data=train,
                          axes=self.axes['particles_train'])
            self.figs['particles_train'].suptitle(
                'SWF, iteration %04d' % epoch)
            self.updated += ['particles_train']

        if 'particles_test' in self.figs and test is not None:
            plot_function(data=test,
                          axes=self.axes['particles_test'])
            self.figs['particles_test'].suptitle(
                'Pre-trained SWF, iteration %04d' % epoch)
            self.updated += ['particles_test']

        if 'closest' in self.figs and match:
            closest = find_closest(vars['particles']['train'][:self.nb_plot],
                                   self.dataset)
            if self.decode_fn is not None:
                closest = self.decode_fn(closest)
            plot_function(data=closest,
                          axes=self.axes['closest'])
            self.figs['closest'].suptitle(
                'Closest samples, iteration %04d' % epoch)
            self.updated += ['closest']

        if 'projections' in self.figs:
            pass
            # sketcher = vars['target_stream']
            # part = vars['particles']['train']
            # module = vars['projector_modules'][42]
            # import ipdb; ipdb.set_trace()
            # sketcher(module, part)

        self.save_figs(filename='%04d' % epoch)
        self.plots_to_purge = newplots
