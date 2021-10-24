from transformer_general import GeneralTransformer as DataTransformer
# from transformer import DataTransformer
import pandas as pd
import numpy as np
from utils_data import get_loaders, generate_synthetic
import torch
from architectures import Encoder, Decoder, Encoder_wae
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from utils_loss import loss_function, score_dataset
import os
import scipy.io as spio
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,rand
import hyperopt as hyperopt


CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"


def train(beta, regularizer, dataset, anom_frac):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embedding_dim = 128
    compress_dims = [128, 128]
    decompress_dims = [128, 128]
    l2scale = 1e-5
    batch_size = 500
    epochs = 10  # 300
    loss_factor = 2

    train_loader, test_loader, val_loader, transformer, data_dim, target_feat_idx, categorical_columns, ordinal_columns, n_samples = get_loaders(
        dataset, anom_frac, batch_size)

    if regularizer == 'kld':
        encoder = Encoder(data_dim, compress_dims, embedding_dim).to(device)
    elif regularizer == 'mmd':
        encoder = Encoder_wae(data_dim, compress_dims, embedding_dim).to(device)
    decoder = Decoder(embedding_dim, decompress_dims, data_dim).to(device)
    optimizerAE = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=l2scale)
    model_reset(decoder,encoder)
    best_auc_val = 0
    for epoch in range(epochs):
        train_loss = 0
        for id_, data in enumerate(train_loader):
            optimizerAE.zero_grad()
            real = data[0].to(device)
            mu, std, logvar = encoder(real)
            if regularizer == 'kld':
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = decoder(emb)
            elif regularizer == 'mmd':
                rec, sigmas = decoder(mu)
            loss_1, loss_2 = loss_function(rec, real, sigmas, mu, logvar, transformer.output_info, loss_factor, beta, regularizer)
            loss = loss_1 + loss_2
            loss.backward()
            optimizerAE.step()
            decoder.sigma.data.clamp_(0.01, 1.0)
            train_loss += loss.item()
        # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
        scores_test, labels_test = score_dataset(encoder, decoder, device, test_loader, transformer.output_info, regularizer)
        AUC_test = roc_auc_score(labels_test, scores_test)
        AUC_test = max(AUC_test, 1-AUC_test)
        # print("AUC_test is %.4f" % (AUC_test))
        scores_val, labels_val = score_dataset(encoder, decoder, device, val_loader, transformer.output_info, regularizer)
        brent_score=brent_measure(scores_test,labels_test)
        AUC_val = roc_auc_score(labels_val, scores_val)
        AUC_val = max(AUC_val, 1-AUC_val)
        # print("AUC_val is %.4f" % (AUC_val))
        if AUC_val > best_auc_val:
            best_auc_val = AUC_val
            final_auc_test = AUC_test
            # df_mapped = generate_synthetic(decoder, n_synthetic, batch_size, embedding_dim, transformer, meta, target_feat_idx, categorical_columns, ordinal_columns, device)
            # filename = "synthetic_data_" + dataset + "_anom_frac_" + '%.2f' % anom_frac + "_regularizer_" + regularizer + "_beta_" + '%.6f' %beta + ".csv"
            # df_mapped.to_csv("./data/simulated/" + filename)
    print("Final AUC for anom_frac %.2f and beta %.5f is %.3f" % (anom_frac, beta, final_auc_test))
    return best_auc_val , AUC_val,brent_score,final_auc_test

def model_reset(decoder,encoder):
    decoder.weight_reset()
    encoder.weight_reset()

def brent_measure(scores_test,labels_test):
    inlier_score=np.mean(scores_test[labels_test==0])
    outlier_score=np.mean(scores_test[labels_test==1])
    return((inlier_score/outlier_score))

def train_valid(params):
    beta_val = params['x']

    regularizer='kld'
    anom_frac=0.1
    best_auc_val,AUC_val,brent_score, final_auc_test= train(beta_val, regularizer, dataset, anom_frac)
    print(brent_score)
    print(AUC_val)
    return (brent_score)


def main(dataset, regularizer, seed):
    #beta_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    #anom_frac_list = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    #auc_array = np.zeros((len(anom_frac_list), len(beta_list)))

    fspace = {
        'x': hp.loguniform('x', -6, -1  )
    }    
    bopt = fmin(fn=train_valid, space=fspace, algo=tpe.suggest, max_evals=20)
    print(bopt)
   

    return(bopt)
    #auc_array[i] = train(beta, regularizer, dataset, anom_frac)

  



if __name__ == '__main__':
    dataset = 'UNSW'
    for regularizer in ['kld']:
        print("Computing results for regularizer ", regularizer)
        for seed in [10]:
            print("seed ", seed)
            print("\n")
            optbeta=main(dataset, regularizer, seed)
            print(optbeta)

            
            