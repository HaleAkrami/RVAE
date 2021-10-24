import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(frac_anom, seed, regularizer):
    dir = './results/simulations/'
    file_name = 'results_frac_anom_%.2f_seed_%d_regularizer_%s' % (frac_anom, seed, regularizer)

    with open(dir + file_name + '.pkl', "rb") as input_file:
        loaded_results = pickle.load(input_file)
    return loaded_results


frac_anom_list = np.array([0, 0.05, 0.1, 0.15, 0.2])
seed_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
regularizer_list = ['mmd', 'kld']

prauc_best_beta_kl_avg = np.zeros_like(frac_anom_list)
prauc_best_beta_mmd_avg = np.zeros_like(frac_anom_list)
prauc_0_beta_kl_avg = np.zeros_like(frac_anom_list)
prauc_0_beta_mmd_avg = np.zeros_like(frac_anom_list)

prauc_best_beta_kl_std = np.zeros_like(frac_anom_list)
prauc_best_beta_mmd_std = np.zeros_like(frac_anom_list)
prauc_0_beta_kl_std = np.zeros_like(frac_anom_list)
prauc_0_beta_mmd_std = np.zeros_like(frac_anom_list)

for i, frac_anom in enumerate(frac_anom_list):
    for regularizer in regularizer_list:
        prauc_best_beta = []
        prauc_0 = []
        for seed in seed_list:
            loaded_results = load_results(frac_anom, seed, regularizer)
            prauc_best_beta.append(loaded_results['best_prauc'])
            prauc_0.append(loaded_results['PRAUC_beta_0'])
        prauc_best_beta = np.array(prauc_best_beta)
        prauc_0 = np.array(prauc_0)
        if regularizer == 'kld':
            prauc_best_beta_kl_avg[i] = np.mean(prauc_best_beta)
            prauc_0_beta_kl_avg[i] = np.mean(prauc_0)
            prauc_best_beta_kl_std[i] = np.std(prauc_best_beta)/(len(seed_list)**0.5)
            prauc_0_beta_kl_std[i] = np.std(prauc_0)/(len(seed_list)**0.5)
        if regularizer == 'mmd':
            prauc_best_beta_mmd_avg[i] = np.mean(prauc_best_beta)
            prauc_0_beta_mmd_avg[i] = np.mean(prauc_0)
            prauc_best_beta_mmd_std[i] = np.std(prauc_best_beta)/(len(seed_list)**0.5)
            prauc_0_beta_mmd_std[i] = np.std(prauc_0)/(len(seed_list)**0.5)


plt.figure()
plt.plot(frac_anom_list, prauc_best_beta_kl_avg, marker='*', markersize=10, lw=3, label=r"$\beta$ + KL", color="black")
plt.fill_between(frac_anom_list,
                 prauc_best_beta_kl_avg-prauc_best_beta_kl_std,
                 prauc_best_beta_kl_avg+prauc_best_beta_kl_std,
 	             facecolor='black', alpha=0.5)

plt.plot(frac_anom_list, prauc_best_beta_mmd_avg, marker='*', markersize=10, lw=3, label=r"$\beta$ + MMD", color="green")
plt.fill_between(frac_anom_list,
                 prauc_best_beta_mmd_avg-prauc_best_beta_mmd_std,
                 prauc_best_beta_mmd_avg+prauc_best_beta_mmd_std,
 	             facecolor='green', alpha=0.5)

plt.plot(frac_anom_list, prauc_0_beta_kl_avg, marker='*', markersize=10, lw=3, label=r"KL + KL", color="red")
plt.fill_between(frac_anom_list,
                 prauc_0_beta_kl_avg-prauc_0_beta_kl_std,
                 prauc_0_beta_kl_avg+prauc_0_beta_kl_std,
 	             facecolor='red', alpha=0.5)

plt.plot(frac_anom_list, prauc_0_beta_mmd_avg, marker='*', markersize=10, lw=3, label=r"KL + MMD", color="blue")
plt.fill_between(frac_anom_list,
                 prauc_0_beta_mmd_avg-prauc_0_beta_mmd_std,
                 prauc_0_beta_mmd_avg+prauc_0_beta_mmd_std,
 	             facecolor='blue', alpha=0.5)

plt.grid(True)
plt.xticks(frac_anom_list)
plt.tick_params(labelsize=12)
plt.xlabel('Injected outliers to training data', fontsize=22)
plt.ylabel('PRAUC', fontsize=22)
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('./figures/prauc_vs_frac_anom.png')