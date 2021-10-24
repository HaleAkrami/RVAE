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
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

##########initialize parameters##########
seed = 10004
epochs = 40
batch_size = 120
log_interval = 10
beta_val = 0.009#0.005  #0.009  #for gaussian use beta=0.006, for bernoulli use beta=0.009
CODE_SIZE = 2
Anomalies = 1
pretrained = 0
SIGMA = 0.2  # for Gaussian Loss function

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
#np.random.seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

#torch.manual_seed(seed)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
############################################



##########load MNIST##########
(X, _), (x_test, y_test) = mnist.load_data()
print(y_test[1:10])
X = X / 255
X = X.astype(float)
X = X[:60000, :, :].astype('float64')
x_test = x_test / 255
x_test = x_test[:10000, :, :].astype('float64')
y_test = y_test[:10000].astype('float64')

X = np.clip(X, 0, 1)
X = np.float64(X > 0.5)

x_test = np.clip(x_test, 0, 1)
x_test=np.float32(x_test>0.5)

if Anomalies == 1:
    N = np.random.rand(6000, 28, 28)
    X[:6000, :, :] = N
    X = np.clip(X, 0, 1)
    X = np.float64(X > 0.5)
    N = np.random.rand(1000, 28, 28)
    indx2 = x_test.shape[0]
    x_test[x_test.shape[0] - 1000:x_test.shape[0], :, :] = N
    y_test[x_test.shape[0] - 1000:x_test.shape[0]] = 10


x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# create train, validation and test data. These are used for computing errors
X_train, X_valid = train_test_split(X, test_size=0.33, random_state=10003)
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))

input = torch.from_numpy(X_train).float()
input = input.to('cuda') if args.cuda else input.to('cpu')

validation_data = torch.from_numpy(X_valid).float()
validation_data = validation_data.to(
    'cuda') if args.cuda else validation_data.to('cpu')

#test_data = torch.from_numpy(x_test).float()
#test_data = test_data.to('cuda') if args.cuda else test_data.to('cpu')

test_data_all = []
for i in range(len(x_test)):
    test_data_all.append(
        [torch.from_numpy(x_test[i]).float(),y_test[i]])

train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                                batch_size=batch_size,
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data_all,
                                          batch_size=batch_size,
                                          shuffle=False)
##############################


##########define model##########
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
        

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        #f sum(sum(torch.isnan(mu)))>0:
        #print(z )
        L=torch.sum(self.decode(z))
        if torch.isnan(L):
            print('nan')
        return self.decode(z), mu, logvar

##############################

##########load model##########
if pretrained == 0:
    model = RVAE().to(device)
else:
    model = torch.load('MNIST_50')
    model = model.to(device)
    model.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#############################

##########define loss########
def MSE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret, 1)
    return ret


def Gaussian_CE_loss(Y, X, beta, sigma=SIGMA):  # 784 for mnist
    D = Y.shape[1]
    term1 = -((1 + beta) / beta)
    K1 = 1 / pow((2 * math.pi * (sigma**2)), (beta * D / 2))
    term2 = MSE_loss(Y, X)
    term3 = torch.exp(-(beta / (2 * (sigma**2))) * term2)
    loss1 = torch.sum(term1 * (K1 * term3 - 1))
    return loss1


def Bernoulli_CE_loss(Y, X, beta):
    term1 = (1 / beta)
    Y = Y* 0.99999 + 1e-6 
    term2 = (X * torch.pow(Y, beta)) + (1 - X) * torch.pow((1 - Y), beta)
    term2 = torch.prod(term2, dim=1) - 1
    term3 = torch.pow(Y, (beta + 1)) + torch.pow((1 - Y), (beta + 1))
    term3 = torch.prod(term3, dim=1) / (beta + 1)
    loss1 = torch.sum((-term1 * term2 + term3) * (beta + 1))
    if torch.isnan(loss1):
        print(loss1)
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def bdiv_elbo(recon_x, x, mu, logvar, beta):

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = Bernoulli_CE_loss(recon_x, x, beta)
        #BBCE = Gaussian_CE_loss(recon_x, x, beta)
    else:
        # if beta is zero use binary cross entropy
        BBCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        #BBCE = torch.sum(MSE_loss(recon_x, x))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD

##########################



##########train model##########
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):

        data = (data).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = bdiv_elbo(recon_batch, data, mu, logvar, beta=beta_val)
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

#############################


##########validation#########
def Validation(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += bdiv_elbo(recon_batch,
                                   data,
                                   mu,
                                   logvar,
                                   beta=beta_val).item()
            #data=torch.tensor(data>0.1)
            recon_batch=torch.tensor((recon_batch>0.5),dtype=torch.float32)
            recon_batch=(recon_batch).to(device)
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
################################


##########test###########
def test(frac_anom):
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

            test_loss_anom += F.binary_cross_entropy(recon_batch* anom_lab, data* anom_lab, reduction='sum').item() 
            test_loss_total +=  F.binary_cross_entropy(recon_batch, data,  reduction='sum').item()

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
    print('====> Test set loss anom: {:.4f}'.format(test_loss_anom))
    print('====> Test set loss normal: {:.4f}'.format(test_loss_normals))


    return test_loss_anom, test_loss_normals
  
#########################



if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        logvar_all, mu_all = Validation(epoch)
        test_loss_anom, test_loss_normals = test(epoch)
       
        with torch.no_grad():
            sample = torch.randn(64, CODE_SIZE).to(device)
            #            sample = .5*torch.eye(20).to(device)

            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')


    #embedding = PCA(n_components=2)
    #Np_mu_transformed = embedding.fit_transform(np.float64(Np_mu))
    #print(Np_mu_transformed.shape)

    
