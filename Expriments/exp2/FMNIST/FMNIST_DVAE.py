from __future__ import print_function
import argparse
import torch
import math
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import scipy.io as spio
#beta = 0.005  #0.00005
#batch_size = 133

seed = 10004
epochs = 150  # was 20
batch_size = 120
log_interval = 10
#beta_val = 0.005  # 0.005 #0.00005,  0.03, 0.005
CODE_SIZE = 20  # was 9
SIGMA = 1.0  # for Gaussian Loss function

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=20,
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

device = torch.device("cuda" if args.cuda else "cpu")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def create_data(frac_anom):

    torch.manual_seed(seed)
    np.random.seed(seed)

    (X, X_lab), (test, _test_lab) = fashion_mnist.load_data()
    X_lab = np.array(X_lab)

    # find bags
    ind = np.isin(X_lab, (0, 1, 2, 3, 4, 5, 6, 8))  #(1, 5, 7, 9)
    X_lab_outliers = X_lab[ind]
    X_outliers = X[ind]

    # find sneaker and ankle boots
    ind = np.isin(X_lab, (7, 9))  # (0, 2, 3, 4, 6))  #
    X_lab = X_lab[ind]
    X = X[ind]
    X = X / 255.0


    Nsamp = np.int(np.rint(len(X) * frac_anom)) + 1

    X_outliers = X_outliers / 255.0
    X[:Nsamp, :, :] = X_outliers[:Nsamp, :, :]
    X_lab[:Nsamp] = 10

    X = np.clip(X, 0, 1)

    # find bags
    ind = np.isin(_test_lab, (0, 1, 2, 3, 4, 5, 6, 8))  #(1, 5, 7, 9)
    _test_lab_outliers = _test_lab[ind]
    test_outliers = test[ind]

    # find sneaker and ankle boots
    ind = np.isin(_test_lab, (7, 9))  # (0, 2, 3, 4, 6))  #
    _test_lab = _test_lab[ind]
    test = test[ind]

    # X = ((X / 255.0) > 0.01).astype(np.float)
    test = test / 255.0
    Nsamp = np.int(np.rint(len(X) * frac_anom)) + 1
    test_outliers = test_outliers / 255.0

    test[:Nsamp, :, :] =test_outliers[:Nsamp, :, :]
    _test_lab[:Nsamp] = 10

    test = np.clip(test, 0, 1)

    X_train, X_valid, X_lab_train, X_lab_valid = train_test_split(
        X, X_lab, test_size=0.33, random_state=10003)
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))
    test = test.reshape((len(test), np.prod(test.shape[1:])))

    train_data = []
    for i in range(len(X_train)):
        train_data.append(
            [torch.from_numpy(X_train[i]).float(), X_lab_train[i]])

    val_data = []
    for i in range(len(X_valid)):
        val_data.append(
            [torch.from_numpy(X_valid[i]).float(), X_lab_valid[i]])

    test_data = []
    for i in range(len(test)):
        test_data.append(
            [torch.from_numpy(test[i]).float(),_test_lab[i]])        

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                              batch_size=len(val_data),
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=len(test_data),
                                              shuffle=False)                                          

    return train_loader, val_loader, test_loader



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


model = RVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def se_loss(Y, X):
    loss1 = torch.sum((X - Y)**2)
    return loss1


def MSE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret)
    return ret


def Gaussian_CE_loss(Y, X, beta, sigma=SIGMA):  # 784 for mnist
    D = Y.shape[1]
    term1 = -((1 + beta) / beta)
    K1 = 1 / pow((2 * math.pi * (sigma**2)), (beta * D / 2))
    term2 = MSE_loss(Y, X)
    term3 = torch.exp(-(beta / (2 * (sigma**2))) * term2)
    loss1 = torch.sum(term1 * (K1 * term3 - 1))
    return loss1


def BBFC_loss(Y, X, beta):
    term1 = (1 / beta)
    term2 = (X * torch.pow(Y, beta)) + (1 - X) * torch.pow((1 - Y), beta)
    term2 = torch.prod(term2, dim=1) - 1
    term3 = torch.pow(Y, (beta + 1)) + torch.pow((1 - Y), (beta + 1))
    term3 = torch.prod(term3, dim=1) / (beta + 1)
    loss1 = torch.sum(-term1 * term2 + term3)
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = Gaussian_CE_loss(recon_x, x.view(-1, 784), beta)
    else:
        # if beta is zero use binary cross entropy
        BBCE = MSE_loss(recon_x.view(-1, 784), x.view(-1, 784))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


def train(epoch, std_dev=1.0):
    model.train()
    train_loss = 0
    for batch_idx, (data, data_lab) in enumerate(train_loader):
        data = (data).to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(
            data + std_dev *
            torch.randn(data.size()).to(device))  # add noise to the input
        loss = beta_loss_function(recon_batch, data, mu, logvar,
                                  beta=0)  # make beta = 0
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


def test(frac_anom,std_dev):
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

            test_loss_anom += MSE_loss(recon_batch * anom_lab,
                                      data * anom_lab).item()
            test_loss_total += MSE_loss(recon_batch, data).item()

            if i == 0:
                n = min(data.size(0), 100)
                samp=[4, 14, 50, 60, 25, 29, 32, 65]
                comparison = torch.cat([
                    data.view(len(recon_batch), 1, 28, 28)[samp],
                    recon_batch.view(len(recon_batch), 1, 28, 28)[samp]
                ])
                save_image(comparison.cpu(),
                           'KBS_results/fashion_mnist_recon_dvae_' +
                           str(std_dev) + '_' + str(frac_anom) + '.png',
                           nrow=n)

        np.savez('KBS_results/fashion_mnist_' + str(std_dev) + '_' +
                 str(frac_anom) + '_dvae.npz',
                 recon=recon_batch.cpu(),
                 data=data.cpu(),
                 anom_lab=anom_lab.cpu())

    test_loss_normals = (test_loss_total - test_loss_anom) / (
        len(test_loader.dataset) - num_anom)
    test_loss_anom /= num_anom
    test_loss_total /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss_total))

    return test_loss_total, test_loss_anom, test_loss_normals

def val(frac_anom, std_dev):
    model.eval()
    test_loss_total = 0
    test_loss_anom = 0
    num_anom = 0
    with torch.no_grad():
        for i, (data, data_lab) in enumerate(val_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            anom_lab = data_lab == 10
            num_anom += np.sum(anom_lab.numpy())  # count number of anomalies
            anom_lab = (anom_lab[:, None].float()).to(device)

            test_loss_anom += MSE_loss(recon_batch * anom_lab,
                                      data * anom_lab).item()
            test_loss_total += MSE_loss(recon_batch, data).item()

            if i == 0:
                n = min(data.size(0), 100)
                #samp = np.arange(200)  # [
                #1 - 1, 101 - 1, 5 - 1, 7 - 1, 15 - 1, 109 - 1, 120 - 1,
                #   26 - 1, 30 - 1, 33 - 1
                # ]  #np.arange(200) #[4, 14, 50, 60, 25, 29, 32, 65]
                samp=[4, 14, 50, 60, 25, 29, 32, 65]
                comparison = torch.cat([
                    data.view(len(recon_batch), 1, 28, 28)[samp],
                    recon_batch.view(len(recon_batch), 1, 28, 28)[samp]
                ])
                save_image(comparison.cpu(),
                           'KBS_results/val_fashion_mnist_recon_dvae' +
                           str(std_dev) + '_' + str(frac_anom) + '.png',
                           nrow=n)

        np.savez('KBS_results/val_fashion_mnist_' + str(std_dev) + '_' +
                 str(frac_anom) + '_dvae.npz',
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


    srange = np.arange(0,1.05,0.1)
    erange = range(1, epochs + 1)
    anrange = np.array([
        
        
        0.1])
   
    test_loss_total = np.zeros((len(anrange), len(srange)))
    test_loss_anom = np.zeros((len(anrange), len(srange)))
    test_loss_normals = np.zeros((len(anrange), len(srange)))

    val_loss_total = np.zeros((len(anrange), len(srange)))
    val_loss_anom = np.zeros((len(anrange), len(srange)))
    val_loss_normals = np.zeros((len(anrange), len(srange)))



    for b, std_dev in enumerate(srange):

        for a, frac_anom in enumerate(anrange):
            train_loader, val_loader, test_loader = create_data(frac_anom)
            model_reset()
            for epoch in erange:

                train(epoch, std_dev=std_dev)

                print('epoch: %d, std_dev=%g, frac_anom=%g' %
                      (epoch, std_dev, frac_anom))

            # save the model
            torch.save(
                model, 'fashion_mnist_beta_dvae_' +
                str(std_dev) + '_frac_anom_' + str(frac_anom))



            test_loss_total[a, b], test_loss_anom[a, b], test_loss_normals[
                a, b] = test(frac_anom, std_dev=std_dev)

            val_loss_total[a, b], val_loss_anom[a, b], val_loss_normals[
                a, b] = val(frac_anom, std_dev=std_dev)


        np.savez('KBS_results/val_loss_fashionmnist_beta_dvae' + str(b) + '.npz',
                val_loss_total=val_loss_total,
                val_loss_anom=val_loss_anom,
                val_loss_normals=val_loss_normals,
                brange=srange,
                anrange=anrange)


        np.savez('KBS_results/test_loss_fashionmnist_beta_dvae' + str(b) + '.npz',
                 test_loss_total=test_loss_total,
                 test_loss_anom=test_loss_anom,
                 test_loss_normals=test_loss_normals,
                 brange=srange,
                 anrange=anrange)




