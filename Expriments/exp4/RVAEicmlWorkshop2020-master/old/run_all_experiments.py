from __future__ import print_function
import argparse
import torch
from architectures import VAE, WAE
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import generate_categorical_data
from sklearn.metrics import average_precision_score
import numpy as np
from utils import cross_entropy, beta_divergence_all_cat, kld, mmd, evaluate_model, train_epoch, compute_loss_ratio, score_dataset

import pickle


def main(frac_anom, seed, regularizer):

    dir = './results/simulations/'
    file_name = 'results_frac_anom_%.2f_seed_%d_regularizer_%s' %(frac_anom, seed, regularizer)

    #################################################################################
    # data parameters
    n_train = 1000
    n_test = 1000
    n_valid = 1000
    C = [10]
    q = len(C)
    mu_anom = 2 #4
    mu_outlier = 2
    rho = 1 #0.25

    #################################################################################
    # network parameters
    torch.manual_seed(seed)
    batch_size = 128
    input_dim = sum(C[:])
    hidden_dim = 12
    latent_dim = 2
    n_epochs = 20
    beta_list = np.logspace(-.01, -8, num=20) #[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    beta_list = np.concatenate([np.zeros(1), beta_list])
    #################################################################################
    # logging parameters
    log_interval = 1000000

    # TODO: change this to gpu later
    device = torch.device("cpu")

    #################################################################################
    # prepare data
    train_loader, test_normal_loader, test_outlier_loader, valid_normal_loader, valid_outlier_loader = generate_categorical_data.main(n_train, n_test, n_valid,  q, mu_anom, mu_outlier, rho, frac_anom, C, batch_size, seed)

    #################################################################################
    loss_ratio_list = np.zeros_like(np.array(beta_list))
    min_val = 1e8
    best_prauc_val = -1e8
    best_prauc_val_0 = -1e8
    PRAUCval_list_all = []
    PRAUC_list_all = []
    loss_ratio_list_all = []

    # define model
    for beta in beta_list:
        if regularizer == 'mmd':
            model = WAE(input_dim, hidden_dim, latent_dim, C).to(device)
        else:
            model = VAE(input_dim, hidden_dim, latent_dim, C).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        #################################################################################
        PRAUCval_list = []
        PRAUC_list = []
        loss_ratio_list = []

        for epoch in range(n_epochs):
            train_epoch(epoch, model, train_loader, C,  beta, regularizer, optimizer, device, log_interval)
            loss_ratio = compute_loss_ratio(model, valid_normal_loader, valid_outlier_loader, device, 0)
            loss_ratio_list.append(loss_ratio)
            PRAUC = evaluate_model(model, test_normal_loader, test_outlier_loader, device, cross_entropy)
            PRAUC_list.append(PRAUC)
            PRAUCval = evaluate_model(model, valid_normal_loader, valid_outlier_loader, device, cross_entropy)
            PRAUCval_list.append(PRAUCval)
            if PRAUCval > best_prauc_val: # beta!=0 and loss_ratio < min_val:
                min_val = loss_ratio
                best_beta = beta
                best_prauc = PRAUC
                best_prauc_val = PRAUCval
                best_epoch = epoch
                torch.save(model, dir + '/models/' + file_name + '_best_beta_model')
            if beta==0 and PRAUCval > best_prauc_val_0:
                PRAUC_beta_0 = PRAUC
                loss_ratio_0 = loss_ratio
                best_prauc_val_0 = PRAUCval
                best_epoch_0 = epoch
                torch.save(model, dir + '/models/' + file_name + '_best_0_model')
        print("For beta %.8f, PRAUC is %.2f, loss ratio is %.2f, PRAUCval is %.2f" %(beta, PRAUC, loss_ratio, PRAUCval) )
        PRAUCval_list_all.append(PRAUCval_list)
        PRAUC_list_all.append(PRAUC_list)
        loss_ratio_list_all.append(loss_ratio_list)

    print("\n")
    print("For beta %d, PRAUC is %.2f, loss ratio is %.2f, PRAUCval is %.2f, best epoch is %d" %(0, PRAUC_beta_0, loss_ratio_0, best_prauc_val_0, best_epoch_0) )
    print("For best beta %.8f, PRAUC is %.2f, loss ratio is %.2f, PRAUCval is %.2f, best epoch is %d" %(best_beta, best_prauc, min_val, best_prauc_val, best_epoch) )

    results = {'best_beta': best_beta,
               'best_prauc': best_prauc,
               'PRAUC_beta_0': PRAUC_beta_0,
               'beta_list': beta_list,
               'best_epoch': best_epoch,
               'best_epoch_0': best_epoch_0,
               'PRAUCval_list_all': PRAUCval_list_all,
               'PRAUC_list_all': PRAUC_list_all,
               'loss_ratio_list_all': loss_ratio_list_all,
               'train_loader': train_loader,
               'test_normal_loader': test_normal_loader,
               'test_outlier_loader': test_outlier_loader,
               'valid_normal_loader': valid_normal_loader,
               'valid_outlier_loader': valid_outlier_loader}


    test_normal_loader, test_outlier_loader, valid_normal_loader, valid_outlier_loader
    with open(dir + file_name + '.pkl', "wb") as output_file:
        pickle.dump(results, output_file)

    # with open(dir + file_name + '.pkl', "rb") as input_file:
    #     loaded_results = pickle.load(input_file)

if __name__ == '__main__':
    frac_anom_list =[0, 0.05, 0.1, 0.15, 0.2] #[0.001, 0.01, 0.1, 0.15, 0.2]
    seed_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    regularizer_list = ['mmd', 'kld']

    for regularizer in regularizer_list:
        for frac_anom in frac_anom_list:
            for seed in seed_list:
                print("Running regularizer %s, frac_anom %.3f, seed %d" %(regularizer, frac_anom, seed))
                main(frac_anom, seed, regularizer)


