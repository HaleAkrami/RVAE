from __future__ import print_function
import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
import argparse
import h5py
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.signal
from sklearn.model_selection import train_test_split

pret=0

def show_and_save(file_name,img):
    f = "/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/figs4/%s.png" % file_name
    save_image(img[2:3,:,:],f)
    
    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)
    
def save_model(epoch, encoder, decoder, D):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    torch.save(D.cpu().state_dict(), 'VAE_GAN_D_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()
    D.cuda()
    
def load_model(epoch, encoder, decoder, D,loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    #D.load_state_dict(torch.load(loc+'/VAE_GAN_D_%d.pth' % epoch))
    #D.cuda()

#####read data######################
d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/data_24_ISEL_100.npz')
X = d['data']

X_data = X[:, :, :, 0:3]
max_val=np.max(X)
#max_val=np.max(max_val,1)
#max_val=np.reshape(max_val,(-1,1,1,3))
X_data = X_data/ max_val
X_data = X_data.astype('float64')
X_valid,X_train, = train_test_split(X, test_size=0.875, random_state=10002,shuffle=False)
X=X_valid
D=X_data.shape[1]*X_data.shape[2]
####################################

#########calculate-Wavlet###########
shape = X_data.shape
max_lev = 1     # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots


fig, axes = plt.subplots(1, 1, figsize=[14, 8])
c = pywt.wavedec2(X_data, 'db2', mode='periodization', level=max_lev,axes=(1, 2))
#c[0] == tuple([np.zeros_like(v) for v in c[0]])
arr_temp, slices = pywt.coeffs_to_array(c,axes=(1, 2))
arr_temp[:,0:64,0:64,:]=0

##normalize##
max_val=np.max(arr_temp)
#arr_final= arr/ np.max(arr_temp)
###################################

##########train validation split##########
batch_size=8


X_valid = np.transpose(X_data, (0, 3, 1,2))
validation_data_inference = torch.from_numpy(X_valid).float()
validation_data_inference= validation_data_inference.to('cuda') 


Validation_loader_inference = torch.utils.data.DataLoader(validation_data_inference,
                                          batch_size=batch_size,
                                          shuffle=False)
                                         
############################################


##########define network##########
class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, representation_size = 64):
        super(Encoder, self).__init__()
        # input parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.features = nn.Sequential(
            # nc x 128x 128
            nn.Conv2d(self.input_channels, representation_size, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size),
            nn.ReLU(),
            # hidden_size x 64 x 64
            nn.Conv2d(representation_size, representation_size*2, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 2),
            nn.ReLU(),
            # hidden_size*2 x 32 x 32
            nn.Conv2d(representation_size*2, representation_size*4, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 4),
            nn.ReLU())
            # hidden_size*4 x 16x 16
            
        self.mean = nn.Sequential(
            nn.Linear(representation_size*4*16*16, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
        self.logvar = nn.Sequential(
            nn.Linear(representation_size*4*16*16, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
    def forward(self, x):
        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mean = self.mean(hidden_representation.view(batch_size, -1))
        logvar = self.logvar(hidden_representation.view(batch_size, -1))

        return mean, logvar
    
    def hidden_layer(self, x):
        batch_size = x.size()[0]
        output = self.features(x)
        return output

class Decoder(nn.Module):
    def __init__(self, input_size, representation_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        
            # 256 x 16 x 16
        self.deconv1 = nn.ConvTranspose2d(representation_size[0], 256, 5, stride=2, padding=2)
        self.act1 = nn.Sequential(nn.BatchNorm2d(256),
                                  nn.ReLU())
            # 256 x 32 x 32
        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
        self.act2 = nn.Sequential(nn.BatchNorm2d(128),
                                  nn.ReLU())
            # 128 x 64 x 64
        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2)
        self.act3 = nn.Sequential(nn.BatchNorm2d(32),
                                  nn.ReLU())
            # 32 x 128 x 128
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
            # 3 x 128 x 128
        self.activation = nn.Tanh()
            
    
    def forward(self, code):
        bs = code.size()[0]
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])
        output = self.deconv1(preprocessed_codes, output_size=(bs, 256, 32, 32))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 128, 64, 64))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 32, 128, 128))
        output = self.act3(output)
        output = self.deconv4(output, output_size=(bs, 3, 128, 128))
        output = self.activation(output)
        return output
class VAE_GAN_Generator(nn.Module):
    def __init__(self, input_channels, hidden_size, representation_size=(256, 16, 16)):
        super(VAE_GAN_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        
        self.encoder = Encoder(input_channels, hidden_size)
        self.decoder = Decoder(hidden_size, representation_size)
        
    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp_()
        
        reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size))).cuda()

        reparametrized_noise = mean + std * reparametrized_noise

        rec_images = self.decoder(reparametrized_noise)
        
        return mean, logvar, rec_images


class Discriminator(nn.Module):
    def __init__(self, input_channels, representation_size=(256, 16, 16)):  
        super(Discriminator, self).__init__()
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.lth_features = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LeakyReLU(0.2))
        
        self.sigmoid_output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        output = self.sigmoid_output(lth_rep)
        return output
    
    def similarity(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        return lth_rep


#################################

########## intilaize parameters##########        
# define constant
input_channels = 3
hidden_size = 64
max_epochs = 200
lr = 3e-4
beta = 0
device='cuda'
#########################################
epoch=49
#HM='/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/figs'
LM='/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/figs11_beta0_ISEL'
##########call network##########
#D = Discriminator(input_channels).cuda()
#G = VAE_GAN_Generator(input_channels, hidden_size).cuda()
################################

##########load low res net##########
G=VAE_GAN_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,D,LM)
#load_model(epoch,G.encoder, G.decoder,D,HM)


##########define optmizer##########
#opt_enc = optim.Adam(G.parameters(), lr=lr)
#opt_dec = optim.Adam(G.decoder.parameters(), lr=lr)
##################################




##########define beta loss##########

def MSE_loss(Y, X):
    ret = (X- Y) ** 2
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1



# Reconstruction + KL divergence losses summed over all elements and batch

def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), beta,sigma,128*128*3)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 128*128*3),x.view(-1, 128*128*3)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

####################################

##########TEST##########
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            seg = X[ind:ind + batch_size, :, :, 3]
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]

            

            f_data = data[:, 2, :, :]
            #f_recon_batch = recon_batch[:, 2, :, :]
            rec_error = f_data - f_recon_batch
            if i<100:
                n = min(f_data.size(0), 100)
                err=(f_data.view(batch_size, 1, 128, 128)[:n] -
                     f_recon_batch.view(batch_size, 1, 128, 128)[:n])
                median=(err).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
                scale_error=np.max(median,axis=2)
                scale_error=np.max(scale_error,axis=2)
                scale_error=np.reshape(scale_error,(-1,1,1,1))
                err=median/scale_error
                err=torch.from_numpy(err)
                err=(err).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 128, 128)[:n],
                    f_recon_batch.view(batch_size, 1, 128, 128)[:n],
                    err,
                    torch.abs(
                        f_data.view(batch_size, 1, 128, 128)[:n] -
                        f_recon_batch.view(batch_size, 1, 128, 128)[:n]),
                    seg.view(batch_size, 1, 128, 128)[:n]
                ])
                save_image(comparison.cpu(),
                           'results4/reconstruction_b' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    
    