import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

cwd = os.getcwd()

betas = np.array([0.00085, 0.0008])
#betas = np.array([0, 0.01])

frac_anom = np.array([ 0.1])
# frac_anom = np.array([0.05])

FPRs = dict()
TPRs = dict()
AUC = dict()

lgd = {
 
    0: 'RAE_0.00065-10%',
    1: 'RAE_0.0008-10%',

}
colors = {0: 'r', 1: 'b'}
lsty = {0: '-', 1: '-'}
c = 0

for beta in betas:
    for frac in frac_anom:
        # beta = 0.01
        # frac = 0.05
        if beta == 0:
            beta_str = f"{beta:.0f}"
        else:
            beta_str = str(beta)

        filename = filename1 = '/ImagePTE1/akrami/git_repos_old/lesion-detector/src/AutoEncoder/L12/original_code/RobustAutoencoder-master/experiments/Outlier Detection/l21out16080_.npz'
        #filename='test_loss_letters_mnist_betas_fracanom.npz'
        #filename = cwd + '/results/faces_' + beta_str + '_' + str(frac) + '.npz'
        # print(filename)

        s = np.load(filename)



        y = s['L']
        x = s['data']
        L = s['anom_lab']
        L[L!=10]=0
        L[L==10]=1

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
plt.ylim([0.0, 1.0])
plt.legend()
plt.show()

fig.savefig('ROCn_RAE.png')

print(AUC)