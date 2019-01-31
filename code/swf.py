# imports
import os
import torch
from torch import nn
from torch import optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import sketch
import data
from sketch import SketchStream
from percentile import Percentile
import argparse
from time import strftime, gmtime
import socket
import functools
import torch.multiprocessing as mp
import queue
from math import floor
from torchvision import transforms
from autoencoder import AE
from interp1d import Interp1d
from math import sqrt
import utils
import numpy as np
import json
from pathlib import Path
import uuid


def swf(train_particles, test_particles, target_stream, num_quantiles,
        stepsize, regularization, num_epochs,
        device_str, logger, results_path="results"):
    """Starts a Sliced Wasserstein Flow with the train_particles, to match
    the distribution whose sketches are given by the target queue.

    The function gets sketches from the queue, and then applies steps of a
    SWF to the particles. The flow is parameterized by a stepsize and a
    regularization parameter. """
    # get the device
    device = torch.device(device_str)

    # prepare stuff
    criterion = nn.MSELoss()
    quantiles = torch.linspace(0, 100, num_quantiles).to(device)
    particles = {}
    particles['train'] = train_particles.to(device)
    if test_particles is not None:
        particles['test'] = test_particles.to(device)
    data_shape = train_particles[0].shape

    # pre-allocate variables
    step = {}
    step_weight = {}
    interp_q = {}
    transported = {}
    particles_qf = {}
    projections = {}
    loss = {}
    train_losses = []
    test_losses = []
    interp_q['train'] = None
    transported['train'] = None
    if test_particles is not None:
        interp_q['test'] = None
        transported['test'] = None

    # call the logger with the transported train and test particles
    logger(particles, -1, loss)

    data_queue = target_stream.queue
    # loop over epochs
    for epoch in range(num_epochs):
        next_queue = stream.ctx.Queue()

        # reset the step for both train and test
        step['train'] = 0
        step['test'] = 0
        step_weight['train'] = 0
        step_weight['test'] = 0
        loss['train'] = 0
        loss['test'] = 0

        print('SWF starting epoch', epoch)
        for (target_qf, projector, id) in iter(data_queue.get, None):
            next_queue.put((target_qf, projector, id))
            print('SWF got in for id', id)
            # get the data from the sketching queue
            target_qf = target_qf.to(device)
            projector = projector.to(device)

            # stepsize *= (index+1)/(index+2)
            for task in particles:  # will include test if provided
                # project the particles
                with torch.no_grad():
                    num_particles = particles[task].shape[0]
                    projections[task] = projector(particles[task])

                    # compute the corresponding quantiles
                    percentile_fn = Percentile(num_quantiles, device)
                    particles_qf[task] = percentile_fn(projections[task])

                    # compute the loss: squared error over the quantiles
                    loss[task] += criterion(particles_qf[task], target_qf)

                    # transort the marginals
                    interp_q[task] = Interp1d()(x=particles_qf['train'].t(),
                                                y=quantiles,
                                                xnew=projections[task].t(),
                                                out=interp_q[task])

                    interp_q[task] = torch.clamp(interp_q[task], 0, 100)
                    transported[task] = Interp1d()(x=quantiles,
                                                   y=target_qf.t(),
                                                   xnew=interp_q[task],
                                                   out=transported[task])
                    step_weight[task] += projections[task].shape[1]
                    step[task] += (
                        projector.backward(
                                transported[task].t() - projections[task])
                        .view(num_particles, *data_shape))
            next_queue.put((target_qf, projector, id))

            print('SWF: finish id', id)

        print('SWF restarting stream')
        # restart the stream
        target_stream.restart()

        print('SWF: updating particles')
        # we got all the updates with the sketches. Now apply the steps
        for task in particles:
            # first apply the step
            particles[task] += stepsize/step_weight[task]*step[task]

            # then possibly add the noise if needed
            noise = torch.randn(*particles[task].shape, device=device)
            noise /= sqrt(particles[task].shape[-1])
            particles[task] += regularization * noise

        print('SWF: now plot')
        # Now do some logging / plotting
        loss = logger(particles, epoch, loss)
        train_losses.append(float(loss['train']))
        if test_particles is not None:
            test_losses.append(float(loss['test']))

        """print('SWF writing results')
        # Saving stuff on disk
        params = {
            'train_losses': [float(x) for x in train_losses],
            'test_losses': [float(x) for x in test_losses],
            'args': vars(args),
            'epochs': int(epoch)
        }
        uuids = uuid.uuid4().hex[:6]

        if not os.path.exists(results_path):
            os.mkdir(results_path)

        with open(Path(results_path,  uuids + ".json"), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))"""


    return (
        (particles['train'], particles['test']) if 'test' in particles
        else particles['train'])


def logger_function(particles, index, loss,
                    plot_dir, log_writer,
                    plot_every, match_every, img_shape, data_loader, ae=None):
    """ Logging function."""

    if log_writer is not None:
        for task in loss:
            log_writer.add_scalar('data/%s_loss' % task,
                                  loss[task].item(), index)
    loss_str = 'iteration %d: ' % (index + 1)
    for item, value in loss.items():
        loss_str += item + ': %0.12f ' % value
    print(loss_str)

    match = match_every > 0 and index > 0 and not index % match_every
    plot = index < 0 or (plot_every > 0 and not index % plot_every)
    if not plot and not match:
        return loss

    # set the number of images we want to plot in the grid
    # for each image we add the closest match (so nb_of_images * 2)
    nb_of_images = 320

    # displays generated images and display closest match in dataset
    for task in particles:
        # if we use the autoencoder deocde the particles to visualize them
        if ae is not None:
            decoder = ae.model.decode.to(particles[task].device)
            cur_task = decoder(particles[task][:nb_of_images, ...])
            img_shape = ae.model.input_shape
        # otherwise just use the particles
        else:
            cur_task = particles[task][:nb_of_images, ...]

        if match:
            # create empty grid
            output_viewport = torch.zeros((nb_of_images,)
                                          + cur_task.shape[1:])

            print("Finding closest matches in dataset")
            closest = utils.find_closest(particles[task][:nb_of_images],
                                         data_loader.dataset)
            output_viewport = decoder(closest.to(device))

            # # iterate over the number of images/particles
            # for k in range(img_viewport.shape[0]):
            #     # find closest match between image_k and sketched dataset
            #     ind, mse = utils.compare_image(
            #         img_viewport[k], data_loader.dataset, 1
            #     )
            #     # load closest match
            #     best_match = data_loader.dataset[int(ind)][0].to(particles[task].device)
            #     # decode closest match
            #     best_match_decoded = decoder(best_match)[0][0]
            #     # add images to output grid
            #     output_viewport[k] = best_match_decoded

            pic = make_grid(
                output_viewport.view(-1, *img_shape),
                nrow=16, padding=2, normalize=True, scale_each=True
            )

            if plot_dir is not None:
                # create the temporary folder for plotting generated samp
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                save_image(
                    pic,
                    '{}/{}_image_match_{}.png'.format(
                        plot_dir, task, index
                    )
                )

        output_viewport = cur_task

        pic = make_grid(
            output_viewport.view(-1, *img_shape),
            nrow=16, padding=2, normalize=True, scale_each=True
        )

        if log_writer is not None:
            log_writer.add_image('%s Image' % task, pic, index)

        if plot_dir is not None:
            # create the temporary folder for plotting generated samples
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            save_image(pic,
                       '{}/{}_image_{}.png'.format(plot_dir, task, index))
    return loss


if __name__ == "__main__":
    # create arguments parser and parse arguments
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Flow.')
    parser = data.add_data_arguments(parser)
    parser = sketch.add_sketch_arguments(parser)

    parser.add_argument("--input_dim",
                        help="Dimension of the random input to the "
                             "generative network. If negative, will "
                             "match the data dimension inferred from dataset",
                        type=int,
                        default=100)
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
    parser.add_argument("--log",
                        help="Flag indicating whether or not to log the "
                             "sliced Wasserstein error along iterations.",
                        action="store_true")
    parser.add_argument("--logdir",
                        help="Directory for the logs using tensorboard. If "
                             "not provided, will log to directory `logs`.",
                        default="logs")
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
    parser.add_argument("--particles_type",
                        help="different kinds of particles. should be "
                             "either RANDOM for random particles "
                             "or TESTSET for particles drawn from the test "
                             "set.")
    parser.add_argument("--num_test",
                        help="number of test samples",
                        type=int,
                        default=0)
    parser.add_argument("--test_type",
                        help="different kinds of test options. should be "
                             "either RANDOM or INTERPOLATE.")
    args = parser.parse_args()

    # prepare the torch device (cuda or cpu ?)
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    # load the data
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset
    train_data_loader = data.load_data(
        args.dataset,
        args.root_data_dir,
        args.img_size,
        args.clip_to
    )
    test_data_loader = data.load_data(
        args.dataset,
        args.root_data_dir,
        args.img_size,
        clipto=-1,
        mode='test',
        batch_size=args.num_samples,
        digits=None
    )

    # prepare AE
    if args.ae:
        train_loader = torch.utils.data.DataLoader(
            train_data_loader.dataset,
            batch_size=32,
            shuffle=True
        )

        autoencoder = AE(
            train_data_loader.dataset[0][0].shape,
            device=device,
            bottleneck_size=args.bottleneck_size,
            convolutive=args.conv_ae
        )
        ae_filename = (args.ae_model
                       + '%d' % args.bottleneck_size
                       + ('conv' if args.conv_ae else 'dense')
                       + '%d' % args.img_size
                       + ''.join(e for e in args.dataset if e.isalnum())
                       + '.model')

        print(ae_filename, 'number of bottleneck features:',
              args.bottleneck_size)

        if not os.path.exists(ae_filename) or args.train_ae:
            train_loader = torch.utils.data.DataLoader(
                train_data_loader.dataset,
                batch_size=32,
                shuffle=True
            )
            print('training AE on', device)
            autoencoder.train(train_loader, nb_epochs=1)
            autoencoder.model = autoencoder.model.to('cpu')
            if args.ae_model is not None:
                torch.save(autoencoder.model.state_dict(), ae_filename)
        else:
            print("Model loaded")
            state = torch.load(ae_filename, map_location='cpu')
            autoencoder.model.to('cpu').load_state_dict(state)

        t = transforms.Lambda(lambda x: autoencoder.model.encode_nograd(x))
        train_data_loader.dataset.transform.transforms.append(t)
        test_data_loader.dataset.transform.transforms.append(t)

    data_shape = train_data_loader.dataset[0][0].shape

    # prepare the projectors
    #projectors = sketch.RandomCoders(args.num_thetas, data_shape)
    projectors = sketch.Projectors(args.num_thetas, data_shape)

    # start sketching
    num_workers = max(1, floor((mp.cpu_count()-2)/2))
    target_stream = SketchStream()
    target_stream.start(num_workers,
                        num_sketches=args.num_sketches,
                        dataloader=train_data_loader,
                        projectors=projectors,
                        num_quantiles=args.num_quantiles)

    # generates the train particles
    print('using ', device)
    if args.input_dim < 0:
        input_dim = projectors.data_dim
    else:
        input_dim = args.input_dim

    if args.particles_type.upper() == "RANDOM":
        train_particles = torch.rand(
            args.num_samples,
            input_dim).to(device)
    elif args.particles_type.upper() == "TESTSET":
        for train_particles in test_data_loader:
            break
        train_particles = train_particles[0].to(device)
        train_particles = train_particles.view(args.num_samples, -1)

    # get the initial dimension for the train particles
    train_particles_dim = np.prod(train_particles.shape[1:])

    # generate test particles
    if not args.num_test:
        test_particles = None
    elif args.test_type.upper() == "INTERPOLATE":
        # Create an interpolation between training particles
        nb_interp_test = 8
        nb_test_pic = 100
        interpolation = torch.linspace(0, 1, nb_interp_test).to(device)
        test_particles = torch.zeros(nb_interp_test * nb_test_pic,
                                     train_particles_dim).to(device)
        for id in range(nb_test_pic):
            for id_in_q, q in enumerate(interpolation):
                test_particles[id*nb_interp_test+id_in_q, :] = (
                 q * train_particles[2*id+1] + (1-q)*train_particles[2*id])
    elif args.test_type.upper() == "RANDOM":
        test_particles = torch.randn(args.num_test,
                                     train_particles_dim).to(device)
    else:
        raise Exception('test type must be either INTERPOLATE or RANDOM')

    print('Train particles dimension:', train_particles_dim)

    # multiply them by a random matrix if not of the appropriate size
    if train_particles_dim != projectors.data_dim:
        print('Using a dimension augmentation matrix')
        torch.manual_seed(0)
        input_linear = torch.randn(input_dim,
                                   projectors.data_dim).to(device)
        train_particles = torch.mm(train_particles, input_linear)
        if test_particles is not None:
            test_particles = torch.mm(test_particles, input_linear)

    train_particles = train_particles.view(-1, *data_shape)
    if test_particles is not None:
        test_particles = test_particles.view(-1, *data_shape)

    # create the logger
    if args.log:
        log_writer = SummaryWriter(os.path.join(args.logdir,
                                                strftime('%Y-%m-%d-%h-%s-',
                                                         gmtime())
                                                + socket.gethostname()))
    else:
        log_writer = None

    # launch the sliced wasserstein flow
    particles = swf(train_particles, test_particles, target_stream,
                     args.num_quantiles, args.stepsize,
                     args.regularization, args.num_epochs,
                     device_str,
                     functools.partial(logger_function,
                                       plot_dir=args.plot_dir,
                                       log_writer=log_writer,
                                       plot_every=args.plot_every,
                                       match_every=args.match_every,
                                       img_shape=data_shape,
                                       data_loader=train_data_loader,
                                       ae=(None if not args.ae
                                           else autoencoder)))
