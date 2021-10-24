import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, C):
        super(VAE, self).__init__()
        self.C = np.array(C)
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)

        index = 0
        slices = []
        for n_category in self.C:
            split_point = index + n_category
            # print((index, split_point))
            slice = F.softmax(h4[:, index:split_point], 1)
            slices.append(slice)
            index = index + n_category
        slice = F.softmax(h4[:, index:-1], 1)
        slices.append(slice)
        out = torch.cat(slices, dim=1)

        return out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class WAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, C, z_var=2):
        super(WAE, self).__init__()
        self.C = np.array(C)
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.z_var = z_var
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)


    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)

        index = 0
        slices = []
        for n_category in self.C:
            split_point = index + n_category
            # print((index, split_point))
            slice = F.softmax(h4[:, index:split_point], 1)
            slices.append(slice)
            index = index + n_category
        slice = F.softmax(h4[:, index:-1], 1)
        slices.append(slice)
        out = torch.cat(slices, dim=1)

        return out

    def forward(self, x):
        z_tilde = self.encode(x.view(-1, self.input_dim))
        z = self.z_var**0.5*torch.randn_like(z_tilde)
        return self.decode(z), z_tilde, z

