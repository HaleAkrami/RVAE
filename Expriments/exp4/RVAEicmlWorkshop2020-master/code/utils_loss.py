import torch
import math
import numpy as np
from torch.nn.functional import cross_entropy
from torch.nn import functional as F


def inverse_multiquad_kernel(z1, z2, z_var, exclude_diag):
    assert z1.shape == z2.shape
    assert len(z1.shape) == 2
    z_dim = z1.shape[1]
    C = 2*z_dim*z_var
    z11 = z1.expand(z1.shape[0], z1.shape[0], z1.shape[1])
    z22 = z2.expand(z2.shape[0], z2.shape[0], z2.shape[1])
    kernel_matrix = C/(1e-9+C+((z11-z22)**2).sum(2))
    if exclude_diag:
        kernel_sum = kernel_matrix.sum() - kernel_matrix.diag().sum()
    else:
        kernel_sum = kernel_matrix.sum()
    return kernel_sum


def mmd(z_tilde, z, z_var=1):
    assert z_tilde.shape == z.shape
    assert len(z.shape) == 2

    n = z.shape[0]

    output = (inverse_multiquad_kernel(z, z, z_var, exclude_diag=True)/(n*(n-1)) +
              inverse_multiquad_kernel(z_tilde, z_tilde, z_var, exclude_diag=True)/(n*(n-1))-
              2*inverse_multiquad_kernel(z, z_tilde, z_var, exclude_diag=False)/(n*n))
    return output # TODO: double check if it is plus or minus


def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor, beta, regularizer):
    st = 0
    loss = []
    for item in output_info:
        if beta == 0:
            if item[1] == 'tanh':
                ed = st + item[0]
                std = sigmas[st]
                loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            elif item[1] == 'softmax':
                ed = st + item[0]
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed
            else:
                assert 0
        else:
            if item[1] == 'tanh':

                D = 1;
                sigma = sigmas[st]
                ed = st + item[0]

                recon_loss = (-(beta + 1) / beta * (1 / ((2 * math.pi * sigma ** 2) ** (beta * D / 2)) * torch.exp(
                    -beta / (2 * sigma ** 2) * (torch.tanh(recon_x[:, st]) - x[:, st]) ** 2) - 1))
                loss.append(torch.sum(recon_loss))
                st = ed

            elif item[1] == 'softmax':
                ed = st + item[0]
                eps = 1e-8
                max_val = recon_x[:, st:ed].max(dim=1, keepdim=True)[0]
                # max_val2 = torch.argmax(x[:, st:ed], dim=-1)
                probs = F.softmax(recon_x[:, st:ed] - max_val, dim=1)
                single_prob = torch.sum(probs * x[:, st:ed], dim=1)
                single_prob = single_prob + (single_prob < eps) * eps
                part1 = (beta + 1) / (beta) * (single_prob ** beta - 1)
                part2 = (probs ** (beta + 1)).sum(dim=1, keepdims=True)
                loss.append(torch.sum(- part1 + part2))
                st = ed

            else:
                assert 0
    assert st == recon_x.size()[1]

    if regularizer == 'kld':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        reg_term = KLD / x.size()[0]
    elif regularizer == 'mmd':
        z = torch.randn_like(mu)
        reg_term = mmd(mu, z)

    out_loss = sum(loss) * factor / x.size()[0], reg_term
    return out_loss


def compute_score(recon_x, x, output_info, sigmas):
    loss = 0
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            std = sigmas[st]
            loss += (((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2))).view(-1, 1)
            #loss += (torch.log(std) * x.size()[0]) #TODO: do we need this for scoring?
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss += (cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='none')).view(-1, 1)
            st = ed
        else:
            assert 0
    return loss

def compute_score_cell(recon_x, x, output_info, sigmas):
    loss = 0
    loss_all=[]
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            std = sigmas[st]
            loss = (((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2))).view(-1, 1)
            if st==0:
                loss_all=loss+0
            else:
                loss_all=torch.cat((loss,loss_all),dim=1)
            #loss += (torch.log(std) * x.size()[0]) #TODO: do we need this for scoring?
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            loss= (cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='none')).view(-1, 1)
            if st==0:
                loss_all=loss+0
            else:
                loss_all=torch.cat((loss,loss_all),dim=1)
            st = ed
        else:
            assert 0
    return loss_all


def score_dataset(encoder, decoder, device, data_iter, output_info, regularizer):
    scores = []
    labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_iter):
            x = data[0].to(device) # batch_size x num_features
            target = data[1].to(device)  # batch_size x 1
            mu, std, logvar = encoder(x)
            if regularizer == 'kld':
                eps = torch.randn_like(std)
                emb = eps * std + mu
                recon_x, sigmas = decoder(emb)
            elif regularizer == 'mmd':
                recon_x, sigmas = decoder(mu)
            loss = compute_score(recon_x, x, output_info, sigmas) # we want beta=0 for scoring
            scores.append(loss.cpu().numpy().flatten())
            labels.append(target.cpu().numpy().flatten())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
    return scores, labels

def score_dataset_cell(encoder, decoder, device, data_iter, output_info, regularizer):
    scores = []
    labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_iter):
            x = data[0].to(device) # batch_size x num_features
            target = data[1].to(device)  # batch_size x 1
            mu, std, logvar = encoder(x)
            if regularizer == 'kld':
                eps = torch.randn_like(std)
                emb = eps * std + mu
                recon_x, sigmas = decoder(emb)
            elif regularizer == 'mmd':
                recon_x, sigmas = decoder(mu)
            loss = compute_score_cell(recon_x, x, output_info, sigmas) # we want beta=0 for scoring
            scores.append(loss.cpu().numpy())
        scores = np.concatenate(scores)

    return scores