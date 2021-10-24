from sklearn.metrics import average_precision_score
import numpy as np
import torch
import errno
import os
import json
import pandas as pd
from mixedDataset import mixedDatasetInstance
from imageDataset import imageDatasetInstance
from sklearn.model_selection import ShuffleSplit
import torch.nn as nn
from torch.nn import functional as F
import math

# loss functions
def cat_loss(x_hat_slice, x_slice, beta, eps=1e-8):
    x_slice = x_slice.long()
    if beta == 0:
        # use cross-entropy loss
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(x_hat_slice, x_slice)
        # print("Avg loss for cat is", loss.mean())
    else:
        # use beta cross entropy
        max_val = x_hat_slice.max(dim=1, keepdim=True)[0]
        probs = F.softmax(x_hat_slice - max_val, dim=1)
        single_prob = probs.gather(1, x_slice.view(-1, 1))
        single_prob = single_prob + (single_prob < eps) * eps
        part1 = (beta + 1) / (beta) * (single_prob ** beta - 1)
        part2 = (probs ** (beta + 1)).sum(dim=1, keepdims=True)
        loss = (- part1 + part2)
    return loss

def num_loss(x_hat_slice, x_slice, logvar, beta):
    x_slice = x_slice.reshape(x_hat_slice.shape)
    if beta == 0:
        # use mse loss loss
        # dtype_float = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # logvar = torch.zeros(1).type(dtype_float)
        logvar_r = (logvar.exp() + 1e-9).log()
        loss = 0.5 * logvar_r + (x_hat_slice.view([-1] + [1]) - x_slice) ** 2 / (2. * logvar_r.exp() + 1e-9)
        loss = loss.view([-1] + [1])
        # sigma = 1
        # loss_func = nn.MSELoss(reduction='none')
        # loss = loss_func(x_hat_slice, x_slice)/(2*sigma + 1e-8)
        # print("Avg loss for real is", loss.mean())
    else:
        # use beta cross entropy
        D = 1;
        sigma = 1
        loss = (-(beta + 1) / beta * (1 / ((2 * math.pi * sigma ** 2) ** (beta * D / 2)) * torch.exp(
            -beta / (2 * sigma ** 2) * (x_hat_slice - x_slice) ** 2) - 1))
    return loss


def total_loss(x, x_hat, dataset_obj, logvar, beta):
    """
    Combine loss from categorical and continous
    """
    index = 0
    loss = 0
    count_num_col = 0
    n_features = len(dataset_obj.num_cols) + len(dataset_obj.cat_cols)
    for col_name, col_type, feat_size in dataset_obj.feat_info:
        if col_name != dataset_obj.target_feat_name:
            if col_type == 'categ': feat_idx = dataset_obj.cat_name_to_idx[col_name]
            if col_type == 'real': feat_idx = dataset_obj.num_name_to_idx[col_name]
            x_slice = x[:, feat_idx]
            split_point = index + feat_size
            x_hat_slice = x_hat[:, index:split_point]
            index = index + feat_size
            if col_type == 'categ':
                loss += (cat_loss(x_hat_slice, x_slice, beta)).view(-1, 1) # TODO: normalize by number of cat or cardinality
            if col_type == 'real':
                loss += (num_loss(x_hat_slice, x_slice, logvar[0][count_num_col], beta)).view(-1, 1)
                count_num_col += 1
            # if (loss != loss).any():
            #     print("bok")
    return loss


def kld_loss(z):
    mu = z['mu'] + 1e-8
    logvar = (z['logvar'].exp() + 1e-8).log()
    kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # print("kld is", kld)
    return kld.view(-1, 1)


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


def mmd(z_tilde, z, z_var):
    assert z_tilde.shape == z.shape
    assert len(z.shape) == 2

    n = z.shape[0]

    output = (inverse_multiquad_kernel(z, z, z_var, exclude_diag=True)/(n*(n-1)) +
              inverse_multiquad_kernel(z_tilde, z_tilde, z_var, exclude_diag=True)/(n*(n-1))-
              2*inverse_multiquad_kernel(z, z_tilde, z_var, exclude_diag=False)/(n*n))
    return output # TODO: double check if it is plus or minus


def score_dataset(model, device, dataset_obj, data_iter):
    scores = []
    labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_iter):
            x = data[0].to(device) # batch_size x num_features
            target = data[1].to(device)  # batch_size x 1
            x_hat, _, logvar = model(x)
            loss = total_loss(x, x_hat, dataset_obj, logvar, 0) # we want beta=0 for scoring
            scores.append(loss.cpu().numpy().flatten())
            labels.append(target.cpu().numpy().flatten())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
    return scores, labels

def compute_loss_ratio(model, device, dataset_obj, data_iter):
    scores, labels = score_dataset(model, device, dataset_obj, data_iter)
    s_normal = scores[labels==0]
    s_outlier = scores[labels==1]
    return s_normal.mean()/s_outlier.mean()


def adjust_learning_rate(optimizer, epoch, milestones):
    """Sets the learning rate to the initial LR decayed by 10 everytime reach the milestones"""
    initial_lr = optimizer.param_groups[0]['lr']
    if epoch in milestones:
        lr = initial_lr /2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# train
def train_epoch(epoch, dataset_obj, model, train_loader, beta, regularizer, optimizer, device, log_interval, milestones=None):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        x = data[0].to(device) # batch_size x num_features
        target = data[1].to(device)  # batch_size x 1
        optimizer.zero_grad()
        x_hat, z, logvar = model(x)

        loss_term = total_loss(x, x_hat, dataset_obj, logvar, beta)

        if regularizer == 'kld':
            reg_term = kld_loss(z)
        elif regularizer == 'mmd':
            z_tilde = z['z_tilde']
            z = z['z']
            reg_term = mmd(z_tilde, z, 1)
            # z_tilde = output1
            # z = output2
            # reg_term = mmd(z_tilde, z, z_var=model.z_var)

        loss = loss_term.mean() + reg_term.mean()
        # print(" loss_term:", loss_term.mean(), "and reg term: ", reg_term.mean())
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
            print(" loss_term:", loss_term.mean(), "and reg term: ", reg_term.mean())

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    if milestones:
        adjust_learning_rate(optimizer, epoch, milestones)


def create_data_folders(run_stats, path_to_folder):

    """ create folders """

    # path to folder where to save to
    path_saving = path_to_folder + run_stats["name"] + "/"

    # try to create folder if not exists yet
    try:
        os.makedirs(path_saving)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for full dirty dataset
    try:
        os.makedirs(path_saving + "/full/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    # path for train dataset
    try:
        os.makedirs(path_saving + "/train/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for validation dataset
    try:
        os.makedirs(path_saving + "/validation/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for test dataset
    try:
        os.makedirs(path_saving + "/test/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    return path_saving


def load_data(folder_path, batch_size, data_type, anom_frac=None,
              get_data_idxs=False, is_one_hot=False):

    # Get column / feature information
    with open(folder_path + 'cols_info.json') as infile:
        data_load = json.load(infile)

    num_feat_names = data_load['num_cols_names']
    cat_feat_names = data_load['cat_cols_names']
    dataset_type = data_load['dataset_type']
    target_feat_name = data_load['target_feat_name']
    is_target_feat_cat = data_load["is_target_feat_cat"]


    get_indexes_flag = True if get_data_idxs else False

    if data_type == 'train':
        folder_path_set = folder_path + "train"
    elif data_type == 'validation':
        get_indexes_flag = False
        folder_path_set = folder_path + "validation"
    else:
        get_indexes_flag = False
        folder_path_set = folder_path + "test"


    type_load = 'clean'
    #Get data folders
    csv_file_path_all = folder_path + "full/data_{}.csv".format(type_load)
    csv_file_path_instance = folder_path_set + "/data_{}.csv".format(type_load)

    # Get train and test data
    dataset = mixedDatasetInstance(csv_file_path_all,
                                   csv_file_path_instance,
                                   num_feat_names, cat_feat_names,
                                   target_feat_name, is_target_feat_cat,
                                   anom_frac,
                                   get_indexes=get_indexes_flag,
                                   use_one_hot=is_one_hot)

    # dataloaders for back-prop
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True)


    # get outputs from the dataset columns names in order
    X = dataset.df_dataset_instance_standardized.values
    attributes = dataset.df_dataset_instance.columns

    return data_loader, X, dataset, attributes


def create_data_splits(run_stats, data):

    cond_test = (run_stats["train_size"]+run_stats["valid_size"]+run_stats["test_size"])==1.0
    assert cond_test, "dataset size percentages (train; valid; test) must match!"

    splitter = ShuffleSplit(n_splits=1, test_size=(1.0-run_stats["train_size"]), random_state=1)
    train_idxs, test_idxs = [x for x in splitter.split(data)][0]

    test_size_prop = float(run_stats["test_size"]) / (run_stats["valid_size"] + run_stats["test_size"])

    splitter_cv = ShuffleSplit(n_splits=1, test_size=test_size_prop, random_state=1)
    rel_valid_idxs, rel_test_idxs  = [x for x in splitter_cv.split(test_idxs)][0]

    validation_idxs = test_idxs[rel_valid_idxs]
    test_idxs = test_idxs[rel_test_idxs]

    return train_idxs, validation_idxs, test_idxs

def save_datasets(path_saving, df_data, train_idxs, validation_idxs, test_idxs):
    ## train dataset
    # clean data
    df_train = df_data.iloc[train_idxs,:]
    df_train = df_train.reset_index(drop=True)

    # save
    df_train.to_csv(path_saving + "/train/" + "data_clean.csv", index=False)
    df_train_idxs = pd.DataFrame(train_idxs, columns=["original_idxs"])
    df_train_idxs.to_csv(path_saving + "/train/" + "original_idxs.csv", index=False)

    ## validation dataset
    # clean data
    df_validation = df_data.iloc[validation_idxs,:]
    df_validation = df_validation.reset_index(drop=True)

    # save
    df_validation.to_csv(path_saving + "/validation/" + "data_clean.csv", index=False)
    df_validation_idxs = pd.DataFrame(validation_idxs, columns=["original_idxs"])
    df_validation_idxs.to_csv(path_saving + "/validation/" + "original_idxs.csv", index=False)

    ## test dataset
    # clean data
    df_test = df_data.iloc[test_idxs,:]
    df_test = df_test.reset_index(drop=True)

    # save
    df_test.to_csv(path_saving + "/test/" + "data_clean.csv", index=False)
    df_test_idxs = pd.DataFrame(test_idxs, columns=["original_idxs"])
    df_test_idxs.to_csv(path_saving + "/test/" + "original_idxs.csv", index=False)

    ## full dataset
    # save
    df_data.to_csv(path_saving + "/full/" + "data_clean.csv", index=False)