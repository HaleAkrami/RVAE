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
import scipy.io as spio
#beta = 0.005  #0.00005
import math
#batch_size = 133
mat = spio.loadmat(
     '/big_disk/akrami/git_repos_old/rvae/validation/matlab/emnist-letters.mat')
data = mat['dataset']

X_train_f = data['train'][0, 0]['images'][0, 0]
y_train_f = data['train'][0, 0]['labels'][0, 0]
X_test_f = data['test'][0, 0]['images'][0, 0]
y_test_f = data['test'][0, 0]['labels'][0, 0]

seed = 10004
epochs = 20
batch_size = 120
log_interval = 10
#beta_val = 0.005  # 0.005 #0.00005,  0.03, 0.005
CODE_SIZE = 20
numbers = np.arange(1, X_test_f.shape[0])
np.random.seed(seed)
np.random.shuffle(numbers)

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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def create_data(frac_anom):

    torch.manual_seed(seed)
    np.random.seed(seed)

    (X, X_lab), (_test_images, _test_lab) = mnist.load_data()
    X = X / 255
    X = X[:10000, ]
    X_lab = X_lab[:10000, ]

    # test_images = test_images / 255

    Nsamp = np.int(np.rint(len(X) * frac_anom)) + 1
    N = np.random.rand(Nsamp, 28, 28)
    inx = numbers[0:Nsamp]
    inx = inx.astype(int)

    N = X_test_f[inx, :] / 255

    #N=np.ones((10000,28,28))

    #X=np.concatenate((X,N),axis=0)
    X[:Nsamp, :, :] = N.reshape(len(inx), X.shape[1], X.shape[2])
    X_lab[:Nsamp] = 10

    X = np.clip(X, 0, 1)
    X = np.float32(X > 0.5) * 0.99999 + 1e-6  # binarize the images

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
                                              batch_size=len(test_data),
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



# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):
    y = recon_x * 0.99999 + 1e-6
    var=torch.exp( logvar)



    # loss
    k = - beta
    d = 20
    d1 = 1 + d * k + 2 * k
    pi_const=torch.Tensor([math.pi])
    pi_const= (pi_const).to(device)
    marginal_likelihood = torch.sum(x * torch.sub(torch.pow(y, (2 * k) / (1 + k)), 1) / (2 * k) + (1 - x)
                                        * torch.sub(torch.pow(1 - y, (2 * k) / (1 + k)), 1) / (k * 2),dim= 1)
    KL_d1 = torch.prod(torch.pow(2*pi_const, k / (1 + d * k)) * torch.sqrt(d1 / (d1 - 2 * k * var))
                           * torch.exp(torch.mul(mu,mu) * d1 * k / (1 + d * k) / (d1 - 2 * k * var)), dim=1)
    KL_d2 = torch.prod(torch.pow(2 * pi_const * var,
                                  k / (1 + k * d)) * math.sqrt(d1 / (1 + d * k)), dim=1)
    KL_divergence = (KL_d1 - KL_d2) / k / 2
    ml = marginal_likelihood
    marginal_likelihood = torch.mean(marginal_likelihood)
    KL_divergence = torch.mean(KL_divergence)
    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO

    return loss




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


def test(frac_anom, beta_val):
    model.eval()
    test_loss_total = 0
    test_loss_anom = 0
    num_anom = 0
    with torch.no_grad():
        for i, (data, data_lab) in enumerate(test_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #recon_batch = torch.tensor(recon_batch > 0.5).float()
            anom_lab = data_lab == 10
            num_anom += np.sum(anom_lab.numpy())  # count number of anomalies
            anom_lab = (anom_lab[:, None].float()).to(device)

            test_loss_anom += se_loss(recon_batch * anom_lab,
                                      data * anom_lab).item()
            test_loss_total += se_loss(recon_batch, data).item()

            if i == 0:
                n = min(data.size(0), 100)
                samp = [96, 97, 99, 90, 14, 35, 53, 57]
                comparison = torch.cat([
                    data.view(len(recon_batch), 1, 28, 28)[samp],
                    recon_batch.view(len(recon_batch), 1, 28, 28)[samp]
                ])

                save_image(comparison.cpu(),
                           'results/letters_mnist_recon_' + str(beta_val) +
                           '_' + str(frac_anom) + '.png',
                           nrow=n)

        np.savez('results/letters_mnist_' + str(beta_val) + '_' +
                 str(frac_anom) + '.npz',
                 recon=recon_batch.cpu(),
                 data=data.cpu(),
                 anom_lab=anom_lab.cpu())

    test_loss_normals = (test_loss_total - test_loss_anom) / (
        len(test_loader.dataset) - num_anom)
    test_loss_anom /= num_anom
    test_loss_total /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss_total))

    return test_loss_total, test_loss_anom, test_loss_normals


if __name__ == "__main__":

    brange = [0.03,0.020]
    #brange = np.arange(0, 0.021, 0.001)  #[0.02] #

    erange = range(1, epochs + 1)
    anrange = [0.1]  # [0.01, 0.05, 0.1]  #
    #anrange = np.arange(0, 0.11, 0.005)  # fraction of anomalies  #

    test_loss_total = np.zeros((len(anrange), len(brange)))
    test_loss_anom = np.zeros((len(anrange), len(brange)))
    test_loss_normals = np.zeros((len(anrange), len(brange)))

    for b, betaval in enumerate(brange):
        for a, frac_anom in enumerate(anrange):

            train_loader, test_loader = create_data(frac_anom)
            model_reset()
            for epoch in erange:

                train(epoch, beta_val=betaval)

                print('epoch: %d, beta=%g, frac_anom=%g' %
                      (epoch, betaval, frac_anom))




            test_loss_total[a, b], test_loss_anom[a, b], test_loss_normals[
                a, b] = test(frac_anom, beta_val=betaval)

        np.savez('test_loss_letters_mnist_betas_fracanom.npz',
                 test_loss_total=test_loss_total,
                 test_loss_anom=test_loss_anom,
                 test_loss_normals=test_loss_normals,
                 brange=brange,
                 anrange=anrange)
