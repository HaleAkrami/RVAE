import numpy as np
import matplotlib.pyplot as plt
import os


def get_auc_single_seed(dataset, regularizer, seed):
    file_dir = './results'
    file_name = file_dir + "/auc_dataset_" + dataset + "_regularizer_" + regularizer + "_seed_" + str(seed) + ".npy"
    auc = np.load(file_name)
    return auc

def get_all_stats(dataset, regularizer):
    seed_list = [10]
    anom_frac_list = [0, 0.01, 0.03, 0.05, 0.07,0.9, 0.1]
    summary_auc = np.zeros((len(seed_list), len(anom_frac_list), 2))

    for idx, seed in enumerate(seed_list):
        auc_single = get_auc_single_seed(dataset, regularizer, seed)
        summary_auc[idx, :, :] = auc_single[:,0:2]

    mean_rvae = summary_auc[0,:,1]
    #std_auc = np.std(summary_auc, axis=0)/(len(seed_list))

    mean_vae = summary_auc[0,:,0]

    mean_rvae[0]=mean_vae[0]
    return mean_vae,  mean_rvae

datasets = 'kdd' #kdd
regularizer = 'kld'
mean_vae_kld, mean_rvae_kld = get_all_stats(datasets, regularizer)
# mean_vae_mmd, std_vae_mmd, mean_rvae_mmd, std_rvae_mmd = get_all_stats(datasets, 'mmd')

anom_frac_list = [0, 0.01, 0.03, 0.05, 0.07,0.09, 0.1]


file_dir = './figures'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

plt.figure()
plt.grid(True)
plt.errorbar(anom_frac_list, mean_vae_kld, color='red', label='VAE-KL', fmt='--o')
# plt.errorbar(anom_frac_list, mean_vae_mmd, yerr=std_vae_mmd, color='darkred', label='VAE-MMD', fmt='--o')

plt.errorbar(anom_frac_list, mean_rvae_kld, color='blue', label='RVAE-KL', fmt='--o')
# plt.errorbar(anom_frac_list, mean_rvae_mmd, yerr=std_rvae_mmd, color='darkblue', label='RVAE-MMD', fmt='--o')
plt.xlabel('Anomaly Fraction in Training Data')
plt.ylabel('AUC of test data')
plt.legend()
plt.savefig(file_dir + "/auc_" + datasets + "_regularizer_" + regularizer +  ".png")

# beta_list = np.array(beta_list)
# best_beta = beta_list[np.argmax(mean_auc, axis=1)]


# plt.figure()
# plt.grid(True)
# plt.errorbar(anom_frac_list, mean_vae, yerr=std_vae, color='red', label='VAE', fmt='--o')
# plt.errorbar(anom_frac_list, mean_rvae, yerr=std_rvae, color='blue', label='RVAE', fmt='--o')
# plt.xlabel('Anomaly Fraction in Training Data')
# plt.ylabel('AUC of test data')
# plt.legend()
# plt.savefig(file_dir + "/auc_kdd_" + regularizer + ".png")
#
# beta_list = np.array(beta_list)
# best_beta = beta_list[np.argmax(mean_auc, axis=1)]
