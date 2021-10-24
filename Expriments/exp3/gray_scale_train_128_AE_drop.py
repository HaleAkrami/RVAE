from __future__ import print_function
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
import random
import math

pret=0

def show_and_save(file_name,img):
    #f = "/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/figs_test/%s.png" % file_name
    f = "/big_disk/akrami/git_repos_new/rvae_orig/validation/Brain_Imaging/figs_test3/%s.png" % file_name
    save_image(img[2:3,:,:],f)
    
 
def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()

    
def load_model(epoch, encoder, decoder):
    decoder.load_state_dict(torch.load('./VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('./VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    


##########load data##############
batch_size =8
###
#d=np.load('/big_disk/akrami/git_repos_new/rvae_orig/validation/Brain_Imaging/data_119_maryland.npz')
d=np.load('/big_disk/akrami/git_repos_new/rvae_orig/validation/Brain_Imaging/data_119_maryland.npz')
X=d['data']

max_val=np.max(X)
X = X/ max_val
X = X.astype('float64')
D=X.shape[1]*X.shape[2]


X_train, X_valid = train_test_split(X, test_size=0.1, random_state=10002,shuffle=False)
print(np.mean(X_train[0,:,:,0]))
X_train = np.transpose(X_train, (0, 3, 1,2))
X_valid = np.transpose(X_valid , (0, 3, 1,2))

input = torch.from_numpy(X_train).float()
input = input.to('cuda') 

validation_data = torch.from_numpy(X_valid).float()
validation_data = validation_data.to('cuda') 

torch.manual_seed(7)
train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=True)
#####network################

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
        
        
        
    def forward(self, x):
        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mean = self.mean(hidden_representation.view(batch_size, -1))

        return mean
    
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
        mean= self.encoder(x)
    

        rec_images = self.decoder(mean)
        
        return mean,rec_images


##############define parameters############
input_channels = 3
hidden_size = 64
max_epochs = 400
lr = 3e-4
#beta =0.00278753

#beta =0.00000055
beta=0.000585


G = VAE_GAN_Generator(input_channels, hidden_size).cuda()



criterion = nn.BCELoss()
criterion.cuda()

opt_enc = optim.Adam(G.parameters(), lr=lr)

random.seed(8)
fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()
data= next(iter(Validation_loader))
fixed_batch = Variable(data).cuda()
row1=random.randint(32,64)-1
row2=random.randint(32,64)-1
fixed_batch [3,:,row1:row1+5,:]=0
fixed_batch [2,:,row2:row2+5,:]=0

##############RVAE Loss#############
def MSE_loss(Y, X):
    ret = (X- Y) ** 2
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1)) #chaged to mean
    return loss1

# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), beta,sigma,128*128*3)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 128*128*3),x.view(-1, 128*128*3)))

    # compute KL divergence


    return BBCE

if pret==1:
    load_model(499, G.encoder, G.decoder)

#############train Model##############
valid_loss_list, train_loss_list= [], []
for epoch in range(max_epochs):
    train_loss=0
    valid_loss=0
    for data in train_loader:
        batch_size = data.size()[0]

 
        datav = Variable(data).cuda()
        l1=random.randint(1,5)-1
        row1=random.randint(32,64)-1
        datav[l1,:,row1:row1+5,:]=0


        mean,  rec_enc = G(datav)
        beta_err=beta_loss_function(rec_enc, datav, mean, beta) 
        err_enc = beta_err
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        train_loss+=beta_err.item()
    train_loss /= len(train_loader.dataset)
        
    G.eval()
    with torch.no_grad():
        for data in Validation_loader:
            data = Variable(data).cuda()
            mean, valid_rec = G(data)
            beta_err=beta_loss_function(valid_rec, data, mean,beta) 
            valid_loss+=beta_err.item()
        valid_loss /= len(Validation_loader.dataset)


    
    print(valid_loss)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)


    _,rec_imgs = G(fixed_batch)
    torch.save(fixed_batch,'AE_batch')
    show_and_save('Input_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]).cpu(),8))
    show_and_save('rec_epoch_%d.png' % epoch ,make_grid((rec_imgs.data[:,2:3,:,:]).cpu(),8))




save_model(epoch, G.encoder, G.decoder)    
plt.plot(train_loss_list, label="train loss")
plt.plot(valid_loss_list, label="validation loss")
plt.legend()
plt.show()    
    