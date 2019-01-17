# imports
import os
import torch
from torch import nn
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
from math import sqrt
import torch.multiprocessing as mp
import queue
from math import floor
from interp1d import Interp1d
from torchvision import transforms
from autoencoder import AE


def swf(train_particles, test_particles, target_queue, num_quantiles,
        stepsize, regularization,
        device_str, logger):
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
    data_dim = train_particles.shape[-1]

    # batch index init
    index = 0

    # pre-allocate variables, for speedup
    interp_q = {}
    transported = {}
    particles_qf = {}
    projections = {}
    loss = {}
    interp_q['train'] = None
    transported['train'] = None
    if test_particles is not None:
        interp_q['test'] = None
        transported['test'] = None
    while True:
        # get the data from the sketching queue
        target_qf, projector, id = target_queue.get()
        target_qf = target_qf.to(device)
        projector = projector.to(device)

        (num_thetas, data_dim) = projector.shape

        for task in particles:  # will include the test particles if provided
            # project the particles
            projections[task] = torch.mm(projector,
                                         particles[task].transpose(0, 1))

            # compute the corresponding quantiles
            percentile_fn = Percentile(num_quantiles, device)
            particles_qf[task] = percentile_fn(projections[task])


            # import matplotlib.pylab as plt
            # plt.clf()
            # plt.plot(target_qf.cpu().numpy().T,'b')
            # plt.plot(particles_qf[task].cpu().numpy().T,'r')
            # plt.show()

            # import ipdb; ipdb.set_trace()
            # compute the loss: squared error over the quantiles
            loss[task] = criterion(particles_qf[task], target_qf)

            # transort the marginals
            interp_q[task] = Interp1d()(x=particles_qf['train'],
                                        y=quantiles,
                                        xnew=projections[task],
                                        out=interp_q[task])

            interp_q[task] = torch.clamp(interp_q[task], 0, 100)
            transported[task] = Interp1d()(x=quantiles,
                                           y=target_qf,
                                           xnew=interp_q[task],
                                           out=transported[task])
            import numpy as np
            pnp = projector.cpu().numpy()
            particles[task] += (
                stepsize/num_thetas *
                torch.mm(
                    (transported[task] - projections[task]).transpose(0, 1),
                    projector))

        # call the logger with the transported train and test particles
        logger(particles, index, loss)

        # now add the noise for the SWF step
        for task in particles:
            noise = torch.randn(
                particles[task].shape[0],
                data_dim, device=device)
            noise /= sqrt(data_dim)
            particles[task] += regularization * noise #sqrt(stepsize*data_dim) * regularization * noise

        index += 1

        # puts back the data into the Queue if it's not full already
        if not target_queue.full():
            try:
                target_queue.put((target_qf.detach().to('cpu'),
                                  projector.detach().to('cpu'), id),
                                 block=False)
            except queue.Full:
                pass

    return (
        (particles['train'], particles['test']) if 'test' in particles
        else particles['train'])


def logger_function(particles, index, loss,
                    plot_dir, log_writer,
                    plot_every, img_shape, ae=None):
    """ Logging function."""

    if log_writer is not None:
        for task in loss:
            log_writer.add_scalar('data/%s_loss' % task,
                                  loss[task].item(), index)
    loss_str = 'iteration %d: ' % (index + 1)
    for item, value in loss.items():
        loss_str += item + ': %0.12f ' % value
    print(loss_str)

    if plot_every < 0 or index % plot_every:
        return

    # displays generated images
    for task in particles:
        if ae is not None:
            cur_task = ae.model.decode.to(particles[task].device)(particles[task])
            img_shape = ae.model.input_shape
        else:
            cur_task = particles[task]
        pic = make_grid(cur_task[:104, ...].view(-1, *img_shape),
                        nrow=8, padding=2, normalize=True, scale_each=True)
        if log_writer is not None:
            log_writer.add_image('%s Image' % task, pic, index)

        if plot_dir is not None:
            # create the temporary folder for plotting generated samples
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            save_image(pic,
                       '{}/{}_image_{}.png'.format(plot_dir, task, index))


if __name__ == "__main__":
    # create arguments parser and parse arguments
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Flow.')
    parser = data.add_data_arguments(parser)
    parser = sketch.add_sketch_arguments(parser)

    parser.add_argument("--input_dim",
                        help="Dimension of the random input to the "
                             "generative network",
                        type=int,
                        default=100)
    parser.add_argument("--num_samples",
                        help="Number of samples to draw per batch to "
                             "compute the sketch of that batch",
                        type=int,
                        default=3000)
    parser.add_argument("--stepsize",
                        help="Stepsize for the SWF",
                        type=float,
                        default=1)
    parser.add_argument("--regularization",
                        help="Regularization term for the additive noise",
                        type=float,
                        default=1e-5)
    parser.add_argument("--plot_every",
                        help="Number of iterations between each plot."
                             " Negative value means no plot",
                        type=int,
                        default=50)
    parser.add_argument("--plot_dir",
                        help="Output directory for the plots",
                        default="./samples")
    parser.add_argument("--model_file",
                        help="Output file for the model",
                        default="./model.pth")
    parser.add_argument("--log",
                        help="Flag indicating whether or not to log the "
                             "sliced Wasserstein error along iterations.",
                        action="store_true")
    parser.add_argument("--logdir",
                        help="Directory for the logs using tensorboard. If "
                             "not provided, will log to directory `logs`.",
                        default="logs")
    parser.add_argument("--train_ae",
                        help="Force training of the AE.",
                        action="store_true")
    parser.add_argument("--ae_model",
                        help="filename for the autoencoder model")
    args = parser.parse_args()

    # prepare the torch device (cuda or cpu ?)
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    # load the data
    if args.root_data_dir is None:
        args.root_data_dir = 'data/'+args.dataset
    data_loader = data.load_data(args.dataset, args.root_data_dir,
                                 args.img_size, args.clip_to)

    # prepare AE
    ae_encode = True
    if ae_encode:
        train_loader = torch.utils.data.DataLoader(
            data_loader.dataset,
            batch_size=32,
            shuffle=True
        )

        autoencoder = AE(
            data_loader.dataset[0][0].shape,
            device=device,
            bottleneck_size=args.input_dim
        )
        if not os.path.exists(args.ae_model) or args.train_ae:
            train_loader = torch.utils.data.DataLoader(
                data_loader.dataset,
                batch_size=32,
                shuffle=True
            )
            print('training AE on', device)
            autoencoder.train(train_loader, nb_epochs=30)
            autoencoder.model = autoencoder.model.to('cpu')
            if args.ae_model is not None:
                torch.save(autoencoder.model.state_dict(), args.ae_model)
        else:
            print("Model loaded")
            state = torch.load(args.ae_model, map_location='cpu')
            autoencoder.model.to('cpu').load_state_dict(state)

        t = transforms.Lambda(lambda x: autoencoder.model.encode_nograd(x))
        data_loader.dataset.transform.transforms.append(t)

    data_shape = data_loader.dataset[0][0].shape
    # prepare the projectors
    projectors = sketch.Projectors(args.num_thetas, data_shape)

    # start sketching
    num_workers = max(1, floor((mp.cpu_count()-2)/2))
    target_stream = SketchStream()
    target_stream.start(num_workers,
                        num_sketches=args.num_sketches,
                        dataloader=data_loader,
                        projectors=projectors,
                        num_quantiles=args.num_quantiles)

    # generates the train particles
    print('using ', device)
    train_particles = torch.rand(args.num_samples, args.input_dim).to(device)

    # generate test particles
    nb_interp_test = 8
    nb_test_pic = 100
    interpolation = torch.linspace(0, 1, nb_interp_test).to(device)
    test_particles = torch.zeros(nb_interp_test * nb_test_pic,
                                 args.input_dim).to(device)

    for id in range(nb_test_pic):
        for id_in_q, q in enumerate(interpolation):
            test_particles[id*nb_interp_test+id_in_q, :] = (
             q * train_particles[id+1] + (1-q)*train_particles[id])

    # multiply them by a random matrix if not of the appropriate size
    if args.input_dim != projectors.data_dim:
        print('Using a dimension augmentation matrix')
        torch.manual_seed(0)
        input_linear = torch.randn(args.input_dim,
                                   projectors.data_dim).to(device)
        train_particles = torch.mm(train_particles, input_linear)
        test_particles = torch.mm(test_particles, input_linear)

    # create the logger
    if args.log:
        log_writer = SummaryWriter(os.path.join(args.logdir,
                                                strftime('%Y-%m-%d-%h-%s-',
                                                         gmtime())
                                                + socket.gethostname()))
    else:
        log_writer = None

    # launch the sliced wasserstein flow
    particles = swf(train_particles, test_particles, target_stream.queue,
                    args.num_quantiles, args.stepsize,
                    args.regularization,
                    device_str,
                    functools.partial(logger_function,
                                      plot_dir=args.plot_dir,
                                      log_writer=log_writer,
                                      plot_every=args.plot_every,
                                      img_shape=data_shape,
                                      ae=autoencoder))
