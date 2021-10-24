from __future__ import print_function
import math
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
import matplotlib.pyplot as plt
#import matplotlib.axes as ax
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
#beta = 0.0005  #0.00005
#batch_size = 133

seed = 10004
epochs = 20
batch_size = 120
log_interval = 10
beta_val = 0.006  #0.005#0.00005,  0.03, 0.005
CODE_SIZE = 2
Noise = 1
pret = 0
sigma = 0.2

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
                    help='random seed (import matplotlib.pyplot as pltult: 1)')
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

(X, _), (x_test, y_test) = mnist.load_data()
D = X.shape[1] * X.shape[2]
print(y_test[1:10])
X = X / 255
X = X.astype(float)
X = X[:60000, :, :].astype('float64')
x_test = x_test / 255
x_test = x_test[:10000, :, :].astype('float64')
y_test = y_test[:10000].astype('float64')
if Noise == 1:
    N = np.random.rand(6000, 28, 28)
    X[:6000, :, :] = N
    X = np.clip(X, 0, 1)

    N = np.random.rand(1000, 28, 28)
    indx2 = x_test.shape[0]
    x_test[x_test.shape[0] - 1000:x_test.shape[0], :, :] = N
    y_test[x_test.shape[0] - 1000:x_test.shape[0]] = 10
    x_test = np.clip(x_test, 0, 1)
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#else:
#X = X[1000:, :, :]
#x_test=x_test[1000:, :, :]
#y_test=y_test[1000:]

X_train, X_valid = train_test_split(X, test_size=0.33, random_state=10003)
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))

input = torch.from_numpy(X_train).float()
input = input.to('cuda') if args.cuda else input.to('cpu')

validation_data = torch.from_numpy(X_valid).float()
validation_data = validation_data.to(
    'cuda') if args.cuda else validation_data.to('cpu')

test_data = torch.from_numpy(x_test).float()
test_data = test_data.to('cuda') if args.cuda else test_data.to('cpu')

train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                                batch_size=batch_size,
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=False)


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
        if sum(sum(torch.isnan(h1))) > 0:
            print(h1)

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #if torch.isnan(h3 ):
        #print(h3 )
        return torch.sigmoid(self.fc4(h3))
        if torch.isnan(torch.sigmoid(self.fc4(h3))):
            print(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        #f sum(sum(torch.isnan(mu)))>0:
        #print(z )
        return self.decode(z), mu, logvar


if pret == 0:
    model = RVAE().to(device)
else:
    model = torch.load('MNIST_50')
    model = model.to(device)
    model.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#weight_decay = 0.01
#optimizer = optim.Adam(model.parameters(), weight_decay = 0.01)


def MSE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret, 1)
    return ret


def BMSE_loss(Y, X, beta, sigma, D):
    term1 = -((1 + beta) / beta)
    K1 = 1 / pow((2 * math.pi * (sigma**2)), (beta * D / 2))
    term2 = MSE_loss(Y, X)
    term3 = torch.exp(-(beta / (2 * (sigma**2))) * term2)
    loss1 = torch.sum(term1 * (K1 * term3 - 1))
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        # If beta is nonzero, use the beta entropy for Gaussian case
        BBCE = BMSE_loss(recon_x, x, beta, sigma, D)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x, x))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data = (data).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = beta_loss_function(recon_batch, data, mu, logvar, beta=beta_val)
        loss.backward()
        if torch.isnan(loss):
            print(loss)
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def Validation(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += beta_loss_function(recon_batch,
                                            data,
                                            mu,
                                            logvar,
                                            beta=beta_val).item()
            if i == 0:
                n = min(data.size(0), 100)
                comparison = torch.cat([
                    data.view(batch_size, 1, 28, 28)[:n],
                    recon_batch.view(batch_size, 1, 28, 28)[:n]
                ])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png',
                           nrow=n)
                mu_all = mu
                logvar_all = logvar
            else:
                mu_all = torch.cat([mu_all, mu])
                logvar_all = torch.cat([logvar_all, logvar])
    test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return logvar_all, mu_all


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += beta_loss_function(recon_batch,
                                            data,
                                            mu,
                                            logvar,
                                            beta=beta_val).item()
            if i == 0:
                mu_all = mu
                logvar_all = logvar
            else:
                mu_all = torch.cat([mu_all, mu])
                logvar_all = torch.cat([logvar_all, logvar])
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return logvar_all, mu_all


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        logvar_all, mu_all = Validation(epoch)
        logvar_all_test, mu_all_test = test(epoch)
        mu_all_test = mu_all_test.cpu()

        Np_mu = mu_all_test.numpy()

        with torch.no_grad():
            sample = torch.randn(64, CODE_SIZE).to(device)
            #            sample = .5*torch.eye(20).to(device)

            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
    print(np.sum(np.isnan(Np_mu)))

    #embedding = PCA(n_components=2)
    #Np_mu_transformed = embedding.fit_transform(np.float64(Np_mu))
    #print(Np_mu_transformed.shape)
    plt.style.use('classic')
    ax = plt.subplot(111)
    y_test[10] = 10
    #ax = fig.add_subplot(111)
    ax.set_aspect(aspect=1)
    #ax.plot(x, y1)
    #ax.plot(x, y2)
    plt.scatter(np.float64(Np_mu[:, 0]),
                np.float64(Np_mu[:, 1]),
                c=y_test,
                edgecolors='face')
    plt.axis([-10, 10, -10, 10], 'equal')
    #plt.axis.set_aspect(1.0)
    #ax.Axes.set_aspect(aspect=1.0 )
    plt.colorbar()
    #ax.set_aspect(1.0)
    plt.show()
    if pret == 0:
        torch.save(model, 'MNIST_50')