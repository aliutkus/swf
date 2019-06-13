# all the dirty plotting stuff goes here

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import numpy as np
from math import floor
import tqdm
import os
import matplotlib as mpl
import pandas as pd

mpl.rcParams['savefig.pad_inches'] = 0
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def add_plotting_arguments(parser):
    parser.add_argument("--basefilename",
                        help="radical of the filename for exports.",
                        default='')
    parser.add_argument("--plot_every",
                        help="Number of iterations between each plot."
                             " Negative value means no plot",
                        type=int,
                        default=100)
    parser.add_argument("--plot_epochs",
                        help="if provided, will plot only at the specified "
                             "epochs",
                        nargs='+',
                        type=int)
    parser.add_argument("--match_every",
                        help="number of iteration between match with the "
                             "items in the database. Negative means this "
                             "is never checked.",
                        type=int,
                        default=-1)
    parser.add_argument("--plot_dir",
                        help="Output directory for the plots",
                        default="samples")
    parser.add_argument("--plot_num_train",
                        help="Number of training samples to plot",
                        type=int,
                        default=104)
    parser.add_argument("--plot_num_test",
                        help="Number of test samples to plot",
                        type=int,
                        default=104)
    parser.add_argument("--plot_num_features",
                        help="Number of features to consider for density "
                             "plots",
                        type=int,
                        default=2)
    parser.add_argument("--no_density_plot",
                        help="If active, will not do the density plots",
                        action="store_true")
    parser.add_argument("--no_particles_plot",
                        help="If active, will not display the particles",
                        action="store_true")
    parser.add_argument("--no_closest_plot",
                        help="If active, will not display the closest "
                             "particles",
                        action="store_true")
    parser.add_argument("--no_swcost_plot",
                        help="If active, will not display the SW cost over "
                             "iterations",
                        action="store_true")
    return parser


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
        n = data.shape[0]
        if n < 16:
            nrow = 1
        elif n % 16 and not n % 8:
            nrow = 8
        else:
            nrow = 16
        pic = make_grid(
            data,
            nrow=nrow,
            padding=2, normalize=True, scale_each=True
            )
        pic_npy = pic.cpu().numpy()
        newplots = [axes.imshow(
            np.transpose(pic_npy, (1, 2, 0)),
            interpolation='nearest')]
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False)
        plt.axis('off')
    elif len(data.squeeze().shape) == 2 and data.squeeze().shape[1] == 2:
        # it's 2D data
        newplots = axes.plot(data.squeeze().cpu().numpy()[:, 0],
                             data.squeeze().cpu().numpy()[:, 1],
                             markers, markersize=1)
    else:
        newplots = []
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return newplots


class SWFPlot:
    """ some big dirty class with just plotting stuff"""
    def __init__(self, features, dataset, plot_dir,
                 no_density_plot=False, no_particles_plot=False,
                 no_closest_plot=False, no_swcost_plot=False,
                 plot_every=1, plot_epochs=None, match_every=1000,
                 plot_num_train=104, plot_num_test=None,
                 decode_fn=None, make_titles=True,
                 dpi=200, basefilename='', extension='png'):
        """
        Initialize the plotting class

        features: int or list of int,
            number of dimensions to consider, or actual features to consider,
            for the density plots
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
        plot_num_train: int
            number of train samples to plot (use -1 for all)
        plot_num_test: int or None
            number of test samples to plot (use -1 for all). If None, will
            do the same as plot_num_train
        decode_fn: function or None
            if not None, the features are sent there for plotting
        make_titles: boolean
            whether or not to write titles on the figures
        dpi: int
            dpi for the figures
        basefilename: string
            base file name for the exports
        extension: string
            the three letters of the file type for exports

            """

        self.density_plot = not no_density_plot
        self.particles_plot = not no_particles_plot
        self.closest_plot = not no_closest_plot
        self.swcost_plot = not no_swcost_plot

        self.plot_every = plot_every
        self.plot_epochs = plot_epochs
        self.match_every = match_every
        self.ndim = features if isinstance(features, int) else len(features)
        self.features = features
        self.dataset = dataset
        self.plot_dir = plot_dir
        self.decode_fn = decode_fn
        self.plot_num_train = plot_num_train
        self.plot_num_test = (plot_num_test if plot_num_test is not None
                             else plot_num_train)
        self.make_titles = make_titles
        self.extension = extension
        self.basefilename = basefilename
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
            fig = plt.figure(1, dpi=dpi)
            fig.clf()

            # prepare the heat map of the distribution of the data
            train_loader = DataLoader(dataset,
                                      batch_size=min(5000, len(dataset)))
            data = next(iter(train_loader))[0].squeeze().cpu().numpy()
            data = data.reshape((data.shape[0], -1))

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
            self.density_xlim = []
            self.density_ylim = []
            print('Preparing the density plots of figures... may take a while')
            pbar = tqdm.tqdm(total=(self.ndim - 1)*self.ndim/2)
            for row in range(self.ndim - 1):
                row_axes = []
                row_xlim = []
                row_ylim = []
                for col in range(row+1):
                    ax = plt.subplot(self.ndim-1, self.ndim-1,
                                     row*(self.ndim-1)+col+1)
                    with sb.axes_style("whitegrid"):
                        sb.kdeplot(
                            data=data[:, self.features[col]],
                            data2=data[:, self.features[row+1]],
                            gridsize=100, n_levels=self.density_nlevels,
                            shade=True,
                            shade_lowest=False,
                            cmap=self.density_data_palette,
                            ax=ax)
                        ax.tick_params(
                            axis='both',
                            which='both',
                            bottom=True,
                            left=True,
                            right=True,
                            top=True,
                            labelbottom=False,
                            labelleft=False)
                    ax.xaxis.set_major_locator(
                        ticker.LinearLocator(numticks=5))
                    ax.yaxis.set_major_locator(
                        ticker.LinearLocator(numticks=5))
                    ax.tick_params(axis=u'both', which=u'both', length=0)
                    ax.set_axisbelow(True)
                    ax.grid(True, zorder=0)
                    ax.relim()

                    row_xlim += [ax.get_xlim()]
                    row_ylim += [ax.get_ylim()]
                    row_axes += [ax]
                    if self.ndim > 2 and row == self.ndim - 2:
                        plt.text(0.5, -(self.ndim-1)/25.,
                                 'feature %d' % self.features[col],
                                 transform=ax.transAxes,
                                 horizontalalignment='center')
                    pbar.update(1)
                if self.ndim > 2:
                    plt.text(1+(self.ndim-1)/40, 0.5,
                             'feature %d' % self.features[row+1], rotation=90,
                             transform=ax.transAxes,
                             verticalalignment='center')
                self.axes['density'] += [row_axes]
                self.density_xlim += [row_xlim]
                self.density_ylim += [row_ylim]
            if self.make_titles:
                self.density_legend = [
                    mpatches.Patch(
                        color=self.density_data_palette(255),
                        label='Training data'),
                    mpatches.Patch(
                        color=self.density_train_palette(255),
                        label='Generated particles')]
                fig.suptitle('Density plot of particles')
                plt.figlegend(handles=self.density_legend, loc='upper center',
                              frameon=True, fancybox=True, ncol=2,
                              bbox_to_anchor=(0.4, 0.88))
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            self.figs['density'] = fig
            self.updated += ['density']

        def new_fig(num, figsize=(10, 8), display_axes=False, title=False):
            fig = plt.figure(num, figsize=figsize, dpi=dpi)
            fig.clf()
            axes = plt.gca()
            if not display_axes:
                axes.xaxis.set_major_formatter(plt.NullFormatter())
                axes.yaxis.set_major_formatter(plt.NullFormatter())
                axes.tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False)
            if title:
                fig.suptitle('This is a placeholder title')
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            return (fig, axes)
        if self.particles_plot:
            (self.figs['particles_train'],
                self.axes['particles_train']) = new_fig(2, figsize=(10, 4),
                                                        title=self.make_titles)
            self.axes['particles_train'].set_anchor('N')
            (self.figs['particles_test'],
                self.axes['particles_test']) = new_fig(3, figsize=(10, 4),
                                                       title=self.make_titles)
            self.axes['particles_test'].set_anchor('N')
        if self.closest_plot:
            (self.figs['closest'], self.axes['closest']) = new_fig(
                4, figsize=(10, 4))
        if self.swcost_plot:
            self.nswcost_plotted = 0
            (self.figs['swcost'], self.axes['swcost']) = new_fig(
                                                            5, figsize=(11, 4),
                                                            display_axes=True,
                                                            title=False)
            self.swcost = pd.DataFrame({'iteration': [],
                                        'task': [],
                                        'SW loss (dB)': []})
            self.swcost.iteration = self.swcost.iteration.astype(int)
            self.figs['swcost'].set_size_inches(10, 3)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            self.updated += ['swcost']

        self.save_figs('init')

    def save_figs(self, filename):
        # create the folder if it doesn't exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        # now loop over the available figures and plot them all
        for name in self.updated:
            self.figs[name].canvas.draw()
            self.figs[name].savefig(
                os.path.join(self.plot_dir, name+self.basefilename + filename

                             + '.' + self.extension),
                bbox_inches='tight', pad_inches=0)
        self.updated = []

    def clear_plots(self):
        for plot in self.plots_to_purge:
            plot.remove()
        self.plots_to_purge = []

    def log(self, vars, epoch):
        # checking whether we need to match and/or plot
        match = (self.match_every > 0
                 and epoch > 0 and not epoch % self.match_every)
        plot = ((self.plot_epochs is None
                 and (self.plot_every > 0 and not epoch % self.plot_every))
                or (self.plot_epochs is not None and epoch in self.plot_epochs)
                )
        if not plot and not match:
            # nothing to do
            return

        # convert the particles to cpu
        train = vars['particles']['train'].to('cpu')
        test = (vars['particles']['test'].to('cpu')
                if 'test' in vars['particles']
                else None)

        self.clear_plots()
        new_plots = []
        self.updated = []
        if 'density' in self.figs:
            train_plot = train.squeeze().view(train.shape[0], -1)
            # plot the actual values of the particles over density plots,
            # just doing this for train particles
            for row in range(self.ndim - 1):
                for col in range(row+1):
                    ax = self.axes['density'][row][col]
                    children = ax.get_children()
                    with sb.axes_style("whitegrid"):
                        ax.set_xlim(*self.density_xlim[row][col])
                        ax.set_ylim(*self.density_ylim[row][col])
                        sb.kdeplot(
                            data=(train_plot[:, self.features[col]]
                                  .cpu().numpy()),
                            data2=(train_plot[:, self.features[row+1]]
                                   .cpu().numpy()),
                            gridsize=100, n_levels=self.density_nlevels,
                            shade=False,
                            cmap=self.density_train_palette,
                            ax=ax)
                        ax.xaxis.set_major_locator(
                            ticker.LinearLocator(numticks=5))
                        ax.yaxis.set_major_locator(
                            ticker.LinearLocator(numticks=5))
                        ax.tick_params(axis=u'both', which=u'both', length=0)
                        ax.grid(True, zorder=0)
                    new_plots += [p for p in ax.get_children()
                                  if p not in children]
            if self.make_titles:
                self.figs['density'].suptitle(
                    'Density plot of particles, iteration %04d' % epoch)
            self.updated += ['density']
        train = train[:self.plot_num_train]
        if test is not None:
            test = test[:self.plot_num_test]
        if self.decode_fn is not None:
            train = self.decode_fn(train)
            if test is not None:
                test = self.decode_fn(test)
        if 'particles_train' in self.figs:
            new_plots_tmp = plot_function(
                                data=train,
                                axes=self.axes['particles_train'])
            if self.make_titles:
                print(self.axes['particles_train'].get_position(), 'positio')
                self.figs['particles_train'].suptitle(
                    'SWF, iteration %04d' % epoch)
                self.figs['particles_train'].tight_layout(
                    rect=[0, 0, 1, 0.92])
            else:
                self.figs['particles_train'].tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            if len(new_plots_tmp):
                new_plots += new_plots_tmp
                self.updated += ['particles_train']
        if 'particles_test' in self.figs and test is not None:
            new_plots_tmp = plot_function(
                                data=test,
                                axes=self.axes['particles_test'])
            if self.make_titles:
                self.figs['particles_test'].suptitle(
                    'Pre-trained SWF, iteration %04d' % epoch)
                self.figs['particles_test'].tight_layout(
                    rect=[0, 0.03, 1, 0.92])
            else:
                self.figs['particles_test'].tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            if len(new_plots_tmp):
                new_plots += new_plots_tmp
                self.updated += ['particles_test']

        if 'closest' in self.figs and match:
            closest = find_closest(
                        vars['particles']['train'][:self.plot_num_train],
                        self.dataset)
            if self.decode_fn is not None:
                closest = self.decode_fn(closest)
            new_plots_tmp = plot_function(
                                data=closest,
                                axes=self.axes['closest'])
            if self.make_titles:
                self.figs['closest'].suptitle(
                    'Closest samples, iteration %04d' % epoch)
                self.figs['closest'].tight_layout(
                    rect=[0, 0.03, 1, 0.95])
            else:
                self.figs['closest'].tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            if len(new_plots_tmp):
                new_plots += new_plots_tmp
                self.updated += ['closest']
        if 'swcost' in self.figs:
            sketcher = vars['sketcher']
            self.nswcost_plotted += 1
            import copy
            # doing a copy of the module, because we are going to move it to
            # another device, and this is a reference to the module of the
            # project
            swmodule = copy.deepcopy(
                            vars['projector_modules'][
                                np.random.randint(100000)])
            for task in vars['particles']:
                particles = vars['particles'][task].clone().detach()
                with torch.no_grad():
                    particles_qf = sketcher(swmodule.to(particles.device),
                                            particles)
                    target_qf = sketcher[swmodule.cpu()]
                errors = (particles_qf.cpu() - target_qf.cpu())**2
                errors = errors.mean(dim=1).detach().numpy()
                errors = 20*np.log10(errors)
                new_errors = pd.DataFrame(
                        {'iteration': (np.ones(errors.shape)
                                       * epoch).astype(int),
                         'task': [task, ]*len(errors),
                         'SW loss (dB)': errors})
                self.swcost = self.swcost.append(new_errors)
            fig = self.figs['swcost']
            fig.clf()
            ax = fig.gca()

            sb.boxplot(
                x='iteration',
                y='SW loss (dB)',
                hue='task',
                data=self.swcost,
                ax=ax,
                fliersize=1,
                linewidth=0.01,
                palette='Set2',
                )
            ax.autoscale(enable=True, tight=True)
            plt.grid(True)
            ax.tick_params(labelsize=6)
            if self.nswcost_plotted > 50:
                ax.get_xaxis().set_ticks([])
            self.updated += ['swcost']

        self.save_figs(filename='%04d' % epoch)
        self.plots_to_purge = new_plots
