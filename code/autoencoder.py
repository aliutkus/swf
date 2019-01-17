from __future__ import print_function
import argparse
import torch.utils.data
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

class DenseEncoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64):
        super(DenseEncoder, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(np.prod(input_shape), bottleneck_size)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x.view(-1, np.prod(self.input_shape))))

class DenseDecoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64):
        super(DenseDecoder, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(bottleneck_size, np.prod(input_shape))

    def forward(self, x):
        return torch.sigmoid(self.fc1(x)).view(
            -1, self.input_shape[0], self.input_shape[1], self.input_shape[2]
        )


class AutoEncoderModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), bottleneck_size=64):
        super(AutoEncoderModel, self).__init__()
        self.input_shape = input_shape
        self.encode = DenseEncoder(input_shape, bottleneck_size)
        self.decode = DenseDecoder(input_shape, bottleneck_size)

    def encode_nograd(self, x):
        with torch.no_grad():
            return self.encode(x)

    def decode_nograd(self, x):
        with torch.no_grad():
            return self.decode(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class AE(object):
    def __init__(self, input_shape, device, bottleneck_size=64):
        super(AE, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.device = device
        self.criterion = nn.BCELoss()
        self.input_shape = input_shape
        self.model = AutoEncoderModel(
            input_shape=self.input_shape,
            bottleneck_size=self.bottleneck_size
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, data_loader, nb_epochs=10):
        for epoch in range(1, nb_epochs + 1):
            self.model.train()
            train_loss = 0
            for batch_idx, (X, _) in enumerate(data_loader):
                X = X.to(self.device)
                self.optimizer.zero_grad()
                Y = self.model(X)
                loss = self.criterion(Y, X)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

            print('AE train => Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(data_loader.dataset)))

    def test(self, data_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (X, _) in enumerate(data_loader):
                X = X.to(device)
                Y = self.model(X)
                test_loss += self.criterion(Y, X).item()
                if i == 0:
                    n = min(X.size(0), 8)
                    comparison = torch.cat([X[:n],
                                        Y.view(data_loader.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                            'results/reconstruction.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('AE test => Test set loss: {:.4f}'.format(test_loss))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    autoencoder = AE(
        train_loader.dataset[0][0].shape,
        device=device,
        nb_epochs=args.epochs,
    )
    autoencoder.train(train_loader)
    autoencoder.test(test_loader)
    with torch.no_grad():
        sample = torch.randn(32, 32).to(device)
        sample = autoencoder.model.decode(sample).cpu()
        save_image(
            sample.view(32, 1, 28, 28),
            'results/sample.png'
        )
