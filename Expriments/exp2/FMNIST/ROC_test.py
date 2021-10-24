import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

cwd = os.getcwd()
#betas = np.array([0, 0.01])
#betas = np.array([0, 0.01])
betas= np.arange(0.0001,0.1,0.01)

#frac_anom = np.array([0.01, 0.05, 0.1])
frac_anom = np.array([0.1])

FPRs = dict()
TPRs = dict()
AUC = dict()

c = 0

for beta in betas:
    for frac in frac_anom:
        # beta = 0.01
        # frac = 0.05
        if beta == 0:
            beta_str = f"{beta:.0f}"
        else:
            beta_str = str(beta)

        #filename = cwd + '/KBS_results/fashion_mnist_' + beta_str + '_'+str(frac) + '.npz'
        #filename = cwd + '/KBS_results/fashion_mnist_' + str(beta) + '_'+str(frac) + '_dvae.npz'
        filename = cwd + '/KBS_results/cp_fashion_mnist_' + beta_str + '_'+str(frac) + '.npz'
        #filename='test_loss_letters_mnist_betas_fracanom.npz'
        #filename = cwd + '/results/faces_' + beta_str + '_' + str(frac) + '.npz'
        # print(filename)

        s = np.load(filename)

        y = s['recon']
        x = s['data']
        L = s['anom_lab']

        y = y.reshape(y.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        L = L.reshape(L.shape[0], -1)

        mse = np.linalg.norm(x - y, 2, 1, True)
        #fpr, tpr, _ = roc_curve(L, mse)
        mse[mse!=mse]=0
        auc = roc_auc_score(L,mse)

        #FPRs[c] = fpr
        #TPRs[c] = tpr
        AUC[c] = auc

        c = c + 1

        # print(c)

print(AUC)