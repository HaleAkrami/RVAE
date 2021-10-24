import argparse
from utils import load_data, train_epoch, score_dataset, compute_loss_ratio
from VAE import VAE
from WAE import WAE
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import os


parser = argparse.ArgumentParser(description='Experiments for NeurIPS')
parser.add_argument('--dataset', type=str, default='DefaultCredit', help='Pick data you want to run experiments') # DefaultCredit, adult
parser.add_argument('--regularizer', type=str, default='kl', help='.')
parser.add_argument('--seed', type=int, default=10, help='.')
parser.add_argument('--batch_size', type=int, default=150, help='.')
parser.add_argument('--latent-dim', type=int, default=20, help='.')
parser.add_argument('--layer-size', type=int, default=400, help='.')
parser.add_argument('--embedding-size', type=int, default=50, metavar='N', help='size of the embeddings for the categorical attributes')
parser.add_argument("--n_epochs", type=int, default=100, metavar="N", help="number of epochs to run for training")


args = parser.parse_args()
data_folder = './' + args.dataset + '/' + args.dataset + '/'

save_dir = './results/' + args.dataset + '/'
model_dir =  save_dir + '/models/'
file_name = 'results_frac_anom_seed_%d_regularizer_%s' % (args.seed, args.regularizer)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# TODO: change this to gpu later
device = torch.device('cuda')
# Load datasets
train_loader, X_train, dataset_obj, attributes = load_data(data_folder, args.batch_size, data_type='train', anom_frac=0)
val_loader, X_val, _, _ = load_data(data_folder, args.batch_size, data_type='validation')
test_loader, X_test, _, _ = load_data(data_folder, args.batch_size, data_type='test')
#
#
beta = 0.01
model = VAE(dataset_obj, args).to(device)

# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
#                            lr=1e-1, weight_decay=1, momentum=0.9)  # excludes frozen params / layers
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.05, weight_decay=1)  # excludes frozen params / layers
# optimizer = optim.Adam(model.parameters(), lr=1e-1)
best_auc_val = -1e8
AUC_train_list = []
AUC_val_list = []
AUC_test_list = []
loss_ratio_list = []
milestones = [10, 30, 70]

for epoch in range(args.n_epochs):
    print(epoch)
    print('learning rate = ' + str(optimizer.param_groups[0]['lr']))
    train_epoch(epoch, dataset_obj, model, train_loader, beta, 'kld', optimizer, device, 10)
    scores_train, labels_train = score_dataset(model, device, dataset_obj, train_loader)
    print("Avg train score", scores_train.mean())
    scores_val, labels_val = score_dataset(model, device, dataset_obj, val_loader)
    scores_test, labels_test = score_dataset(model, device, dataset_obj, test_loader)
    # PRAUC_train = average_precision_score(labels_train, scores_train)
    AUC_test = roc_auc_score(labels_test, scores_test)
    AUC_val = roc_auc_score(labels_val, scores_val)
    loss_ratio = compute_loss_ratio(model, device, dataset_obj, val_loader)

    # PRAUC_train_list.append(PRAUC_train)
    AUC_val_list.append(AUC_val)
    AUC_test_list.append(AUC_test)

    if AUC_val > best_auc_val:  # beta!=0 and loss_ratio < min_val:
        min_val = loss_ratio
        best_beta = beta
        best_auc_test = AUC_test
        best_auc_val = AUC_val
        best_epoch = epoch
        torch.save(model, model_dir + file_name + '_best_beta_model')

    print("For beta %.8f, AUC_test is %.2f, loss ratio is %.2f, AUCval is %.2f" % (beta, AUC_test, loss_ratio, AUC_val))

print("X")

