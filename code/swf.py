# imports
import os
import torch
from torch import nn
from torchvision.utils import save_image
from torchvision.utils import make_grid
import qsketch
import data
import argparse
import functools
import torch.multiprocessing as mp
import networks
from torchinterp1d import Interp1d
from torchpercentile import Percentile
import utils
from math import sqrt
from tqdm import tqdm, trange


def swf(train_particles, test_particles, target_stream, projector_modules,
        stepsize, regularization, num_epochs,
        device_str, logger, results_path="results"):
    """Starts a Sliced Wasserstein Flow with the train_particles, to match
    the distribution whose sketches are given by the target queue.

    The function gets sketches from the queue, and then applies steps of a
    SWF to the particles. The flow is parameterized by a stepsize and a
    regularization parameter. """

    # get the device
    device = torch.device(device_str)

    # pre-allocate variables
    particles = {}
    particles['train'] = train_particles.to(device)
    if test_particles is not None:
        particles['test'] = test_particles.to(device)

    step = {}
    step_weight = {}
    interp_q = {}
    transported = {}
    particles_qf = {}
    projections = {}
    loss = {}
    for task in particles:
        interp_q[task] = None
        transported[task] = None

    data_shape = train_particles[0].shape
    criterion = nn.MSELoss()
    projector = projector_modules[0].to(device)
    data_queue = target_stream.queue
    percentiles = target_stream.percentiles.clone().to(device)
    bar_epoch = trange(num_epochs, desc="epoch")

    # call the logger with the transported train and test particles
    logger(bar_epoch, locals(), -1)

    # loop over epochs
    for epoch in bar_epoch:
        if target_stream.shared_data['num_epochs'] == 1:
            next_queue = mp.Queue()

        # reset the step for both train and test
        for task in particles:
            step[task] = 0
            step_weight[task] = 0
            loss[task] = 0

        pbar = tqdm(total=target_stream.shared_data['num_sketches'])
        # get the data from the sketching queue until the None sentinel
        for (target_qf, id) in iter(data_queue.get, None):
            # putting back the data to our temporary queue
            if target_stream.shared_data['num_epochs'] == 1:
                next_queue.put((target_qf.detach().clone(), id))

            # putting the target quantiles to device and setting the projector
            target_qf = target_qf.to(device)
            projector.recycle(id)

            for task in particles:  # will include test if provided
                with torch.no_grad():
                    num_particles = particles[task].shape[0]
                    # project the particles
                    projections[task] = projector(particles[task])

                    # compute the corresponding quantiles
                    particles_qf[task] = Percentile()(projections[task],
                                                      percentiles)

                    # compute the loss: squared error over the quantiles
                    loss[task] += criterion(particles_qf[task], target_qf)

                    # transort the marginals, using only the quantiles of train
                    interp_q[task] = Interp1d()(x=particles_qf['train'].t(),
                                                y=percentiles,
                                                xnew=projections[task].t(),
                                                out=interp_q[task])
                    transported[task] = Interp1d()(x=percentiles,
                                                   y=target_qf.t(),
                                                   xnew=interp_q[task],
                                                   out=transported[task])

                    step_weight[task] += projections[task].shape[1]
                    step[task] += (
                        projector.backward(
                                transported[task].t() - projections[task])
                        .view(num_particles, *data_shape))
            pbar.update(1)

        # we got all the updates with the sketches. Now apply the steps
        for task in particles:
            # first apply the step
            particles[task] += stepsize/step_weight[task]*step[task]

            # then possibly add the noise if needed
            noise = torch.randn(*particles[task].shape, device=device)
            noise /= sqrt(particles[task].shape[-1])
            particles[task] += regularization * noise

        # if we are reusing the same sketches again, we prepare the sentinel
        # and replace the data source by what we just recorded
        if target_stream.shared_data['num_epochs'] == 1:
            next_queue.put(None)
            data_queue = next_queue

        # Now do some logging / plotting
        loss = logger(bar_epoch, locals(), epoch)

    return (
        (particles['train'], particles['test']) if 'test' in particles
        else particles['train'])


def plot_function(data, axes, plot_dir, filename):
    import numpy as np
    import matplotlib.pyplot as pl
    data = data.detach().cpu()
    if plot_dir is None or filename is None:
        return
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
        newplots = pl.plot(data.squeeze().numpy()[:, 0],
                           data.squeeze().numpy()[:, 1], 'r.', markersize=1)

    # create the temporary folder for plotting generated samp
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    figure = axes.get_figure()
    figure.canvas.draw()
    figure.savefig(os.path.join(plot_dir, filename))
    for p in newplots:
        p.remove()
    return


def logger_text(writer, vars, index):
    # plain text output
    loss_str = 'epoch %d: ' % (index + 1)
    for item, value in vars['loss'].items():
        loss_str += item + ': %0.12f ' % value
    writer.write(loss_str)


def logger_particles_only(writer, vars, index,
                          plot_dir, plot_every, match_every,
                          dataset, decode_fn, nb_match=300):
    """ Logging function."""
    # first output some text to console
    logger_text(writer, vars, index)

    # checking whether we need to match and/or plot
    match = match_every > 0 and index > 0 and not index % match_every
    plot = index < 0 or (plot_every > 0 and not index % plot_every)
    if not plot and not match:
        return

    # prepares the generated outputs and their closest match in dataset
    # if required
    for task in particles:
        if match:
            # create empty grid
            writer.write("Finding closest matches in dataset")
            closest = utils.find_closest(particles[task][:nb_match],
                                         dataset)
        else:
            closest = None

        # if we use the autoencoder deocde the particles to visualize them
        plot_data = particles[task][:nb_plot]

        if decode_fn is not None:
            plot_data = decode_fn(plot_data.to('cpu'))
        plot_function(data=plot_data,
                      axes=axes,
                      plot_dir=plot_dir,
                      filename='%s_%04d.png' % (task, index))

            if decode_fn is not None:
                closest = decode_fn(closest)
            plot_function(data=closest,
                          axes=axes,
                          plot_dir=plot_dir,
                          filename='%s_match_%04d.png' % (task, index))
    return loss


def logger_particles_decoder(writer, vars, index,
                             plot_dir, plot_every, match_every,
                             dataset, decode_fn, nb_match=300):
    """ Logging function."""
    # first output some text to console
    logger_text(writer, vars, index)

    # checking whether we need to match and/or plot
    match = match_every > 0 and index > 0 and not index % match_every
    plot = index < 0 or (plot_every > 0 and not index % plot_every)
    if not plot and not match:
        return

    # prepares the generated outputs and their closest match in dataset
    # if required
    for task in particles:
        if match:
            # create empty grid
            writer.write("Finding closest matches in dataset")
            closest = utils.find_closest(particles[task][:nb_match],
                                         dataset)
        else:
            closest = None
        for plot_fn in plot_fns:
            plot_fn(particles=particles,
                    closest=closest,
                    index=index,
                    plot_dir=plot_dir,
                    filename='%s_%04d.png' % (task, index))
        for
        # if we use the autoencoder deocde the particles to visualize them
        plot_data = particles[task][:nb_plot]

        if decode_fn is not None:
            plot_data = decode_fn(plot_data.to('cpu'))
        plot_function(data=plot_data,
                      axes=axes,
                      plot_dir=plot_dir,
                      filename='%s_%04d.png' % (task, index))

            if decode_fn is not None:
                closest = decode_fn(closest)
            plot_function(data=closest,
                          axes=axes,
                          plot_dir=plot_dir,
                          filename='%s_match_%04d.png' % (task, index))
    return loss


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # create arguments parser and parse arguments
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Flow.')
    parser = qsketch.add_sketch_arguments(parser)
    parser = data.add_data_arguments(parser)

    parser.add_argument("--input_dim",
                        help="Dimension of the random input to the "
                             "generative network. If negative, will "
                             "match the data dimension inferred from dataset",
                        type=int,
                        default=100)
    parser.add_argument("--num_thetas",
                        help="Number of thetas per sketch.",
                        type=int,
                        default=2000)
    parser.add_argument("--num_epochs",
                        help="Number of epochs",
                        type=int,
                        default=5000)
    parser.add_argument("--bottleneck_size",
                        help="Dimension of the bottleneck features",
                        type=int,
                        default=64)
    parser.add_argument("--num_samples",
                        help="Number of samples to draw per batch to "
                             "compute the sketch of that batch",
                        type=int,
                        default=3000)
    parser.add_argument("--stepsize",
                        help="Stepsize for the SWF",
                        type=float,
                        default=1e-3)
    parser.add_argument("--regularization",
                        help="Regularization term for the additive noise",
                        type=float,
                        default=0)
    parser.add_argument("--no_fixed_sketch",
                        help="If active, will generate new sketch at "
                             "each epoch",
                        action="store_true")
    parser.add_argument("--plot_every",
                        help="Number of iterations between each plot."
                             " Negative value means no plot",
                        type=int,
                        default=100)
    parser.add_argument("--match_every",
                        help="number of iteration between match with the "
                             "items in the database. Negative means this "
                             "is never checked.",
                        type=int,
                        default=-1)
    parser.add_argument("--plot_dir",
                        help="Output directory for the plots",
                        default="samples")
    parser.add_argument("--ae",
                        help="Activate reduction dimension through auto-"
                             "encoding.",
                        action="store_true")
    parser.add_argument("--conv_ae",
                        help="Activate convolutive AE",
                        action="store_true")
    parser.add_argument("--train_ae",
                        help="Force training of the AE.",
                        action="store_true")
    parser.add_argument("--ae_model",
                        help="filename for the autoencoder model")
    parser.add_argument("--num_test",
                        help="number of test samples",
                        type=int,
                        default=0)
    parser.add_argument("--test_type",
                        help="different kinds of test options. should be "
                             "either RANDOM or INTERPOLATE.")
    args = parser.parse_args()

    # prepare the torch device (cuda or cpu ?)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # load the data
    train_data = data.load_image_dataset(
        dataset=args.dataset,
        data_dir=args.root_data_dir,
        img_size=args.img_size
    )

    # prepare AE
    if args.ae:
        autoencoder = networks.AE(
            train_data[0][0].shape,
            device=device,
            bottleneck_size=args.bottleneck_size,
            convolutive=args.conv_ae
        )
        ae_filename = os.path.join(
            'weights',
            args.ae_model
            + '%d' % args.bottleneck_size
            + ('conv' if args.conv_ae else 'dense')
            + '%d' % args.img_size
            + ''.join(e for e in args.dataset if e.isalnum())
            + '.model')

        print(ae_filename, 'number of bottleneck features:',
              args.bottleneck_size)

        if not os.path.exists(ae_filename) or args.train_ae:
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=32,
                shuffle=True
            )
            print('training AE on', device)
            autoencoder.train(train_loader, nb_epochs=20)
            autoencoder.model = autoencoder.model.to('cpu')
            if args.ae_model is not None:
                torch.save(autoencoder.model.state_dict(), ae_filename)
        else:
            state = torch.load(ae_filename, map_location='cpu')
            autoencoder.model.to('cpu').load_state_dict(state)
            print("Model loaded")

        # augmenting the dataset with calling the encoder, to get an item
        train_data = data.TransformedDataset(
            train_data,
            transform=autoencoder.model.encode_nograd)

    data_shape = train_data[0][0].shape

    # Launch the data stream
    data_stream = qsketch.DataStream(train_data)
    data_stream.stream()

    # prepare the sketcher
    target_stream = qsketch.Sketcher(data_source=data_stream,
                                     percentiles=torch.linspace(
                                            0, 100, args.num_quantiles),
                                     num_examples=args.num_examples)

    # prepare the projectors
    projectors = qsketch.ModulesDataset(
                        networks.LinearProjector,
                        shape_in=data_shape,
                        num_out=args.num_thetas)

    target_stream.stream(modules=projectors,
                         num_sketches=args.num_sketches,
                         num_epochs=(
                            args.num_epochs if args.no_fixed_sketch
                            else 1),
                         num_workers=args.num_workers)

    # generates the train particles
    print('using ', device)
    if args.input_dim < 0:
        input_shape = data_shape
    else:
        input_shape = [args.input_dim, ]

    train_particles = torch.randn(
        args.num_samples,
        *input_shape).to(device)

    # get the initial dimension for the train particles
    train_particles_shape = train_particles.shape[1:]

    # generate test particles
    if not args.num_test:
        test_particles = None
    elif args.test_type.upper() == "INTERPOLATE":
        # Create an interpolation between training particles
        nb_interp_test = 8
        nb_test_pic = 100
        interpolation = torch.linspace(0, 1, nb_interp_test).to(device)
        test_particles = torch.zeros(nb_interp_test * nb_test_pic,
                                     *train_particles_shape).to(device)
        for id in range(nb_test_pic):
            for id_in_q, q in enumerate(interpolation):
                test_particles[id*nb_interp_test+id_in_q, :] = (
                 q * train_particles[2*id+1] + (1-q)*train_particles[2*id])
    elif args.test_type.upper() == "RANDOM":
        test_particles = torch.randn(args.num_test,
                                     *train_particles_shape).to(device)
    else:
        raise Exception('test type must be either INTERPOLATE or RANDOM')

    print('Train particles dimension:', torch.tensor(
        train_particles_shape).numpy())

    # multiply them by a random matrix if not of the appropriate size
    if train_particles_shape != data_shape:
        print('Using a dimension augmentation matrix')
        torch.manual_seed(0)
        input_linear = torch.randn(
            torch.prod(torch.tensor(input_shape)),
            torch.prod(torch.tensor(data_shape))).to(device)
        train_particles = torch.mm(
            train_particles.view(args.num_samples, -1),
            input_linear)
        if test_particles is not None:
            test_particles = torch.mm(test_particles, input_linear)

    train_particles = train_particles.view(-1, *data_shape)
    if test_particles is not None:
        test_particles = test_particles.view(-1, *data_shape)

    import matplotlib.pyplot as plt
    if args.bottleneck_size == 2:
        # we will have a 2D plot, so preparing the vizualization
        import seaborn as sb
        test = torch.cat([train_data[id][0] for id in range(50000)], dim=0)
        plt.autoscale(True, tight=True)
        plt.clf()
        sb.set_style(sb.axes_style('whitegrid'))
        plot = sb.kdeplot(
            test[:, 0].numpy(),
            test[:, 1].numpy(),
            gridsize=100, n_levels=200,
            linewidths=0.1,
            clip=((-1, 12), (-1, 10))
            )
        plt.xlabel('feature 1')
        plt.ylabel('feature 2')
        plt.title('Distribution of bottleneck features on MNIST')
        plot.get_figure().savefig('test.pdf')
        plot_axes = plt.gca()
    else:
        plt.clf()
        plot_axes = plt.gca()

    # launch the sliced wasserstein flow
    particles = swf(train_particles=train_particles,
                    test_particles=test_particles,
                    target_stream=target_stream,
                    modules=projectors,
                    stepsize=args.stepsize,
                    regularization=args.regularization,
                    num_epochs=args.num_epochs,
                    device_str=device_str,
                    logger=functools.partial(
                          logger_function,
                          axes=plot_axes,
                          plot_dir=args.plot_dir,
                          plot_every=args.plot_every,
                          match_every=args.match_every,
                          dataset=train_data,
                          nb_plot=-1,
                          decode_fn=None#(
                            # autoencoder.model.decode_nograd if args.ae
                            # else None
                          )
                    )
