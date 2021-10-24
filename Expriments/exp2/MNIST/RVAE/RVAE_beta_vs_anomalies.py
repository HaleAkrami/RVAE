from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
#beta = 0.005  #0.00005
#batch_size = 133

seed = 10004
epochs = 20
batch_size = 120
log_interval = 10
#beta_val = 0.005  # 0.005 #0.00005,  0.03, 0.005
CODE_SIZE = 4

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='enables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(seed)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def create_data(frac_anom):

    (X, X_lab), (_test_images, _test_lab) = mnist.load_data()
    X = X / 255
    X = X[:10000, ]
    X_lab = X_lab[:10000, ]

    # test_images = test_images / 255

    Nsamp = np.int(np.rint(len(X) * frac_anom)) + 1
    N = np.random.rand(Nsamp, 28, 28)
    #N=np.ones((10000,28,28))

    #X=np.concatenate((X,N),axis=0)
    X[:Nsamp, :, :] = N
    X_lab[:Nsamp] = 10

    X = np.clip(X, 0, 1)
    X_train, X_valid, X_lab_train, X_lab_valid = train_test_split(
        X, X_lab, test_size=0.33, random_state=10003)
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))

    train_data = []
    for i in range(len(X_train)):
        train_data.append(
            [torch.from_numpy(X_train[i]).float(), X_lab_train[i]])

    test_data = []
    for i in range(len(X_valid)):
        test_data.append(
            [torch.from_numpy(X_valid[i]).float(), X_lab_valid[i]])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True)

    return train_loader, test_loader


class RVAE(nn.Module):
    def __init__(self):
        super(RVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, CODE_SIZE)
        self.fc22 = nn.Linear(400, CODE_SIZE)
        self.fc3 = nn.Linear(CODE_SIZE, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def weight_reset(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()


#        self.fc1.reset_parameters()
#        self.fc21.reset_parameters()
#        self.fc1.reset_parameters()

model = RVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def se_loss(Y, X):
    loss1 = torch.sum((X - Y)**2)
    return loss1


def BBFC_loss(Y, X, beta):
    term1 = (1 / beta)
    #print(X)
    #print(Y)
    term2 = (X * torch.pow(Y, beta)) + (1 - X) * torch.pow((1 - Y), beta)
    term2 = torch.prod(term2, dim=1) - 1
    #print(term2.shape)
    term3 = torch.pow(Y, (beta + 1)) + torch.pow((1 - Y), (beta + 1))
    term3 = torch.prod(term3, dim=1) / (beta + 1)
    loss1 = torch.sum(-term1 * term2 + term3)
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = BBFC_loss(recon_x, x.view(-1, 784), beta)
    else:
        # if beta is zero use binary cross entropy
        BBCE = F.binary_cross_entropy(recon_x,
                                      x.view(-1, 784),
                                      reduction='sum')

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


def train(epoch, beta_val):
    model.train()
    train_loss = 0
    #    for batch_idx, data in enumerate(train_loader):
    for batch_idx, (data, data_lab) in enumerate(train_loader):
        #    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data = (data).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = beta_loss_function(recon_batch, data, mu, logvar, beta=beta_val)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def model_reset():
    model.weight_reset()


def test(epoch, beta_val):
    model.eval()
    test_loss_total = 0
    test_loss_anom = 0
    num_anom = 0
    with torch.no_grad():
        for i, (data, data_lab) in enumerate(test_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            anom_lab = data_lab == 10
            num_anom += np.sum(anom_lab.numpy())  # count number of anomalies
            anom_lab = (anom_lab[:, None].float()).to(device)

            test_loss_anom += se_loss(recon_batch * anom_lab,
                                      data * anom_lab).item()
            test_loss_total += se_loss(recon_batch, data).item()

            if i == 0:
                n = min(data.size(0), 100)
                comparison = torch.cat([
                    data.view(batch_size, 1, 28, 28)[:n],
                    recon_batch.view(batch_size, 1, 28, 28)[:n]
                ])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(beta_val) + '_' +
                           str(epoch) + '.png',
                           nrow=n)

    test_loss_normals = (test_loss_total - test_loss_anom) / (
        len(test_loader.dataset) - num_anom)
    test_loss_anom /= num_anom
    test_loss_total /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss_total))

    return test_loss_total, test_loss_anom, test_loss_normals


if __name__ == "__main__":

    brange = np.arange(0, 0.02, 0.001)
    erange = range(1, epochs + 1)
    anrange = np.arange(0, 0.2, 0.01) # fraction of anomalies
    test_loss_total = np.zeros((len(anrange), len(brange)))
    test_loss_anom = np.zeros((len(anrange), len(brange)))
    test_loss_normals = np.zeros((len(anrange), len(brange)))

    for b, betaval in enumerate(brange):
        for a, frac_anom in enumerate(anrange):
            train_loader, test_loader = create_data(frac_anom)
            model_reset()
            for epoch in erange:

                train(epoch, beta_val=betaval)

                test_loss_total[a, b], test_loss_anom[a, b], test_loss_normals[
                    a, b] = test(epoch, beta_val=betaval)

                print('epoch: %d, beta=%g, frac_anom=%g' %
                      (epoch, betaval, frac_anom))

        np.savez('test_loss' + str(b) + '.npz',
                 test_loss_total=test_loss_total,
                 test_loss_anom=test_loss_anom,
                 test_loss_normals=test_loss_normals,
                 brange=brange,
                 anrange=anrange)
