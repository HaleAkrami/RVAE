import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

cwd = os.getcwd()
srange = np.arange(0, 1.05, .1)
#betas = np.array([0, 0.01])

frac_anom = [.1]#np.array([0.01, 0.05, 0.1])
# frac_anom = np.array([0.05])

FPRs = dict()
TPRs = dict()
AUC = dict()

lgd = {
    0: 'VAE-1%',
    1: 'VAE-5%',
    2: 'VAE-10%',
    3: 'RVAE-1%',
    4: 'RVAE-5%',
    5: 'RVAE-10%'
}
colors = {0: 'r', 1: 'b', 2: 'k', 3: 'r', 4: 'b', 5: 'k'}
lsty = {0: '--', 1: '--', 2: '--', 3: '-', 4: '-', 5: '-'}
c = 0

for std_dev in srange:
    for frac in frac_anom:


        filename = cwd + '/results/fashion_mnist_' + str(std_dev) + '_denoising_ae_' + str(frac) + '.npz'
        filename = cwd + '/results/letters_mnist_denoising_ae_' + str(std_dev) + '_' + str(frac) + '.npz'
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
        fpr, tpr, _ = roc_curve(L, mse)
        auc = roc_auc_score(L,mse)

        FPRs[c] = fpr
        TPRs[c] = tpr
        AUC[c] = auc

        c = c + 1

        # print(c)

lw = 2
fig = plt.figure(figsize=(8, 6), dpi=300)

for c in np.arange(0, len(FPRs)):
    fpr = FPRs[c]
    tpr = TPRs[c]

    plt.plot(fpr, tpr, lsty[c], color=colors[c], lw=lw, label=lgd[c])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # print(c)

#plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='-.')
plt.xlim([0.0, 1.0])
plt.ylim([0.5, 1.05])
plt.legend()
plt.show()

fig.savefig('ROCn_fashion_mnist_bernoulli_v211.png')

print(AUC)