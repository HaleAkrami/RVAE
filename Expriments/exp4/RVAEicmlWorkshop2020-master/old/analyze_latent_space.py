import pickle
import numpy as np
import matplotlib.pyplot as plt
from analyze_results import load_results
import torch
from utils import cross_entropy, evaluate_model

def get_latent(model, regularizer, data_iter):
    z_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_iter):
            data = data.to(device)
            recon, o1, o2 = model(data)
            if regularizer == 'mmd':
                z = o1
            elif regularizer == 'kld':
                mu, logvar = o1, o2
                # z = o1
                z = model.reparameterize(mu, logvar)
            z_list.append(z)
        z_all = np.concatenate(z_list)
    return z_all


device = torch.device("cpu")

frac_anom = 0
seed = 30
regularizer = 'mmd'


torch.manual_seed(seed)
dir = './results/simulations/'
file_name = 'results_frac_anom_%.2f_seed_%d_regularizer_%s' %(frac_anom, seed, regularizer)
loaded_results = load_results(frac_anom=frac_anom, seed=seed, regularizer=regularizer)

best_beta_model_file_name = dir + '/models/' + file_name + '_best_beta_model'
model_best_beta = torch.load(best_beta_model_file_name)
model_best_beta.eval()

best_0_model_file_name = dir + '/models/' + file_name + '_best_0_model'
model_best_0 = torch.load(best_0_model_file_name)
model_best_0.eval()

outlier_data = loaded_results['test_outlier_loader']
normal_data = loaded_results['test_normal_loader']

z_best_beta_corrupted_normal = get_latent(model_best_beta, regularizer, normal_data)
z_best_0_corrupted_normal = get_latent(model_best_0, regularizer, normal_data)

z_best_beta_corrupted_outlier = get_latent(model_best_beta, regularizer, outlier_data)
z_best_0_corrupted_outlier = get_latent(model_best_0, regularizer, outlier_data)

vmin = min( min(np.min(z_best_beta_corrupted_normal), np.min(z_best_0_corrupted_normal)),
                min(np.min(z_best_beta_corrupted_outlier), np.min(z_best_0_corrupted_outlier)))

vmax = max( max(np.max(z_best_beta_corrupted_normal), np.max(z_best_0_corrupted_normal)),
                max(np.max(z_best_beta_corrupted_outlier), np.max(z_best_0_corrupted_outlier)))


plt.figure()
plt.scatter(z_best_beta_corrupted_normal[:,0], z_best_beta_corrupted_normal[:, 1], color='blue', s=8, label=r'Normal $\beta$ '+regularizer, vmin=vmin, vmax=vmax)
plt.scatter(z_best_beta_corrupted_outlier[:,0], z_best_beta_corrupted_outlier[:, 1], color='green', s=8, label=r'Outlier $\beta$ '+regularizer, vmin=vmin, vmax=vmax)

# plt.legend()
#
# plt.figure()
# plt.scatter(z_best_beta_corrupted_outlier[:,0], z_best_beta_corrupted_outlier[:, 1], color='red', marker='*', s=10, label='beta outlier', vmin=vmin, vmax=vmax)
# plt.legend()
# #
# plt.figure()
plt.scatter(z_best_0_corrupted_normal[:,0], z_best_0_corrupted_normal[:, 1], color='red', s=8, label='Normal KL '+regularizer, vmin=vmin, vmax=vmax)
plt.scatter(z_best_0_corrupted_outlier[:,0], z_best_0_corrupted_outlier[:, 1], color='black', s=8, label='Outlier KL '+regularizer, vmin=vmin, vmax=vmax)
# plt.legend()
#
# plt.figure()
# plt.scatter(z_best_0_corrupted_outlier[:,0], z_best_0_corrupted_outlier[:, 1], color='red', marker='*', s=10, label='kl outlier', vmin=vmin, vmax=vmax)


plt.grid(True)
plt.xlabel(r'$z_1$', fontsize=22)
plt.ylabel(r'$z_2$', fontsize=22)
plt.title("Latent Space representation when regularizer is" + regularizer)
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('./figures/latent_mmd.png')
# PRAUC_best_beta = evaluate_model(model_best_beta, loaded_results['test_normal_loader'],
#                        loaded_results['test_outlier_loader'],
#                        device, cross_entropy)
# print(PRAUC_best_beta)
#
# PRAUC_best_0 = evaluate_model(model_best_0, loaded_results['test_normal_loader'],
#                        loaded_results['test_outlier_loader'],
#                        device, cross_entropy)
# print(PRAUC_best_0)

