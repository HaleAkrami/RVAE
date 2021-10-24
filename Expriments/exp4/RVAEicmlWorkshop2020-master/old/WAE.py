#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F
# from model_utils import nll_categ_global, nll_gauss_global

from EmbeddingMul import EmbeddingMul


class WAE(nn.Module):

    def __init__(self, dataset_obj, args):

        super(WAE, self).__init__()
        # NOTE: for feat_select, (col_name, col_type, feat_size) in enumerate(dataset_obj.feat_info)

        self.dataset_obj = dataset_obj
        self.args = args

        self.size_input = len(dataset_obj.cat_cols)*self.args.embedding_size + len(dataset_obj.num_cols)
        self.size_output = len(dataset_obj.cat_cols) + len(dataset_obj.num_cols) # 2*

        ## Encoder Params
        # define a different embedding matrix for each feature
        self.feat_embedd = nn.ModuleList([nn.Embedding(c_size, self.args.embedding_size, max_norm=1)
                                         for col_name, col_type, c_size in dataset_obj.feat_info
                                         if col_type=="categ" and col_name!=self.dataset_obj.target_feat_name])

        self.fc1 = nn.Linear(self.size_input, self.args.layer_size)
        self.fc21 = nn.Linear(self.args.layer_size, self.args.latent_dim)
        self.fc22 = nn.Linear(self.args.layer_size, self.args.latent_dim)

        ## Decoder Params
        self.fc3 = nn.Linear(self.args.latent_dim, self.args.layer_size)
        self.out_cat_linears = nn.ModuleList([nn.Linear(self.args.layer_size, c_size)
                                              for col_name, col_type, c_size in dataset_obj.feat_info
                                              if col_name != self.dataset_obj.target_feat_name])

        ## Log variance of the decoder for real attributes
        if dataset_obj.num_cols:
            # TODO: why?
            self.logvar_x = nn.Parameter(torch.zeros(1, len(dataset_obj.num_cols)).float())
        else:
            self.logvar_x = []

        ## Other
        self.activ = nn.LeakyReLU(0.2)
        self.logSoftmax = nn.LogSoftmax(dim=1)

        # define encoder / decoder easy access parameter list
        # encoder params
        encoder_list = [self.fc1, self.fc21, self.fc22]
        self.encoder_mod = nn.ModuleList(encoder_list)
        if self.feat_embedd:
            self.encoder_mod.append(self.feat_embedd)
        self.encoder_param_list = nn.ParameterList(self.encoder_mod.parameters())

        # decoder params
        decoder_list = [self.fc3, self.out_cat_linears]
        self.decoder_mod = nn.ModuleList(decoder_list)
        self.decoder_param_list = nn.ParameterList(self.decoder_mod.parameters())
        if len(self.logvar_x):
            self.decoder_param_list.append(self.logvar_x)


    def get_inputs(self, x_data, one_hot_categ=False):
        # mixed data, or just real or just categ
        input_list = []
        cursor_embed = 0
        start = 0
        for col_name, col_type, feat_size in self.dataset_obj.feat_info:
            if one_hot_categ:
                # TODO: did not test
                pass
            else:
                if col_type == "categ" and col_name != self.dataset_obj.target_feat_name: # categorical (uses embeddings)
                    feat_idx = self.dataset_obj.cat_name_to_idx[col_name]
                    input_list.append(self.feat_embedd[cursor_embed](x_data[:, feat_idx].long()))
                    cursor_embed += 1

                elif col_type == "real" and col_name != self.dataset_obj.target_feat_name: # numerical
                    feat_idx = self.dataset_obj.num_name_to_idx[col_name]
                    input_list.append(x_data[:, feat_idx].view(-1, 1))

        return torch.cat(input_list, 1)


    def encode(self, x_data, one_hot_categ=False):
        input_values = self.get_inputs(x_data, one_hot_categ)
        h1 = self.activ(self.fc1(input_values))
        z = {'mu': self.fc21(h1), 'logvar': self.fc22(h1)}
        return z


    def sample_normal(self, z, eps=None):
        if self.training:
            if eps is None:
                eps = torch.randn_like(z['mu'])
            std = z['logvar'].mul(0.5).exp_()
            return eps.mul(std).add_(z['mu'])
        else:
            return z['mu']

    def reparameterize(self, z, eps_samples=None):
        z_samples = self.sample_normal(z, eps_samples)
        return z_samples

    def decode(self, z):
        p_params = dict()
        h3 = self.activ(self.fc3(z))
        out_cat_list = []
        for feat_idx, out_cat_layer in enumerate(self.out_cat_linears):
            if self.dataset_obj.feat_info[feat_idx][1] == "categ": # coltype check
                out_cat_list.append(out_cat_layer(h3))
            elif self.dataset_obj.feat_info[feat_idx][1] == "real":
                out_cat_list.append(out_cat_layer(h3))

        # tensor with dims (batch_size, self.size_output)
        x_recon = torch.cat(out_cat_list, 1)

        if self.dataset_obj.num_cols:
            logvar = self.logvar_x.clamp(-3, 3) #p_params['logvar_x'] = self.logvar_x
        else:
            logvar = None

        return x_recon, logvar

    def forward(self, x_data, n_epoch=None, one_hot_categ=False):
        z = self.encode(x_data, one_hot_categ)
        z_tilde = z['mu']
        # z_samples = self.reparameterize(z)
        x_recon, logvar = self.decode(z_tilde)
        z = torch.randn_like(z_tilde) # assume prior for z is normal Gaussian
        zs = dict()
        zs['z_tilde'] = z_tilde
        zs['z'] = z
        return x_recon, zs, logvar





