# imports
import os
import torch
from torch import nn
from torchvision.utils import save_image
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from qsketch import sketch, data
from qsketch.sketch import SketchStream
from qsketch.autograd_percentile import Percentile
import argparse
from time import strftime, gmtime
import socket
import functools
from math import sqrt
import torch.multiprocessing as mp
import queue
from math import floor
from pytorch_interp1d.interpolate import interp1d


def train_swf(particles, target_queue, num_quantiles,
              stepsize, regularization,
              device_str, logger):

    # get the device
    device = torch.device(device_str)

    # create the flow stream object
    model = SketchStream()

    # prepare stuff
    criterion = nn.MSELoss()
    quantiles = torch.linspace(0, 100, num_quantiles).to(device)
    particles = particles.to(device)
    (num_samples, data_dim) = particles.shape

    # for each sketch
    index = 0
    interp_q = None
    transported = None
    while True:
        target_qf, projector, id = target_queue.get()
        target_qf = target_qf.to(device)
        projector = projector.to(device)

        (num_thetas, data_dim) = projector.shape
        # project the generated output
        projections = torch.mm(projector,
                               particles.transpose(0, 1))

        # compute the corresponding quantiles
        percentile_fn = Percentile(num_quantiles, device)
        particles_qf = percentile_fn(projections)

        # compute the loss: squared error over the quantiles
        loss = criterion(particles_qf, target_qf)

        # transort the marginals
        interp_q = interp1d(x=particles_qf,
                            y=quantiles,
                            xnew=projections,
                            out=interp_q)
        interp_q = torch.clamp(interp_q, 0, 100)
        transported = interp1d(x=quantiles,
                               y=target_qf,
                               xnew=interp_q,
                               out=transported)

        noise = torch.randn(num_samples, data_dim, device=device) / sqrt(data_dim)
        #import ipdb; ipdb.set_trace()
        particles += (stepsize/num_thetas *
                      torch.mm((transported - projections).transpose(0, 1),
                               projector))
        logger(particles, model, index, loss)

        particles += sqrt(stepsize*data_dim) * regularization * noise

        index += 1

        #print('should put to the queue here')
        #model.queue.put((particles_qf.detach().to('cpu'), None, None))

        # puts back the data into the Queue if it's not full already
        if not target_queue.full():
            try:
                target_queue.put((target_qf.detach().to('cpu'),
                                  projector.detach().to('cpu'), id),
                                 block=False)
            except queue.Full:
                pass

    return model


def logger_function(particles, model, index, loss,
                    plot_dir, log_writer,
                    plot_every, img_shape, checkpoint_path):
    if log_writer is not None:
        log_writer.add_scalar('data/loss', loss.item(), index)
    print('iteration {}, loss:{:.6f}'
          .format(index + 1, loss.item()))

    if plot_every < 0 or index % plot_every:
        return

    # save a checkpoint
    #model.dump(checkpoint_path)

    # displays genrated images
    pic = make_grid(particles[:104, ...].view(-1, *img_shape),
                    nrow=8, padding=2, normalize=True, scale_each=True)
    if log_writer is not None:
        log_writer.add_image('Image', pic, index)
    if plot_dir is not None:
        # create the temporary folder for plotting generated samples
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        save_image(pic,
                   '{}/image_{}.png'.format(plot_dir, index))


if __name__ == "__main__":
    # create arguments parser and parse arguments
    parser = argparse.ArgumentParser(description='Learn a generative model '
                                     'through sketching.'
                                     'The model is learned by fitting '
                                     'batches of sketches.')
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

    # while not stream.queue.full():
    #     pass
    # stream.pause()
    # print('now saving the stream')
    # stream.dump('queue.data')

    # generates the particles
    particles = torch.rand(args.num_samples, args.input_dim).to(device)

    # multiply them by a random matrix if not of the appropriate size
    if args.input_dim != projectors.data_dim:
        torch.manual_seed(0)
        input_linear = torch.randn(args.input_dim,
                                   projectors.data_dim).to(device)
        particles = torch.mm(particles, input_linear)

    # create the logger
    if args.log:
        log_writer = SummaryWriter(os.path.join(args.logdir,
                                                strftime('%Y-%m-%d-%h-%s-',
                                                         gmtime())
                                                + socket.gethostname()))
    else:
        log_writer = None

    # launch the sliced wasserstein flow
    particles = train_swf(particles, target_stream.queue,
                          args.num_quantiles, args.stepsize,
                          args.regularization,
                          device_str,
                          functools.partial(logger_function,
                                            plot_dir=args.plot_dir,
                                            log_writer=log_writer,
                                            plot_every=args.plot_every,
                                            img_shape=data_shape,
                                            checkpoint_path=args.model_file))
