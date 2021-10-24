import pandas as pd
import numpy as np
import os
import errno
import json
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy
from utils import create_data_folders, create_data_splits, save_datasets
from sklearn.model_selection import ShuffleSplit


# 1 - Select dataset
dataset = 'adult' #'Wine' # 'Letter'
name_file = "adult.data"
folder_path = "./adult/"


names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
         'occupation', 'relationship', 'race', 'sex', 'capital-gain',
         'capital-loss', 'hours-per-week', 'native-country', 'salary']

df_data = pd.read_csv(folder_path + name_file, names=names)
df_data = df_data.drop(columns=['education', 'fnlwgt'])

# NOTE: we are not using fnlwgt and education as in https://github.com/DPautoGAN/DPautoGAN/blob/master/uci/uci.ipynb
num_feat_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']

cat_feat_names = ['workclass', 'education-num', 'marital-status',
                  'occupation', 'relationship', 'race', 'sex',
                  'native-country']

target_feat_name = ['salary']
is_target_feat_cat = True

train_size = 0.8 # TODO: change this ratio for real training
valid_size = 0.1
run_stats = {"name":dataset, "train_size": train_size, "valid_size": valid_size,
             "cat_cols_names":cat_feat_names, "num_cols_names":num_feat_names, "target_feat_name":target_feat_name,
             "is_target_feat_cat": is_target_feat_cat}

cols_info = {"cat_cols_names": run_stats["cat_cols_names"],
             "num_cols_names": run_stats["num_cols_names"],
             "target_feat_name": run_stats["target_feat_name"],
             "is_target_feat_cat": run_stats["is_target_feat_cat"],
             "dataset_type": "mixed"
             }
path_saving = create_data_folders(run_stats, folder_path)
with open(path_saving + 'cols_info.json', 'w') as outfile:
    json.dump(cols_info, outfile, indent=4, sort_keys=True)

splitter = ShuffleSplit(n_splits=1, test_size=valid_size, random_state=1)
train_idxs = [x for x in splitter.split(df_data)][0][0] # shape 29304
validation_idxs = [x for x in splitter.split(df_data)][0][1] # shape 3256

# save
df_train = df_data.iloc[train_idxs, :]
df_train = df_train.reset_index(drop=True)
df_train.to_csv(path_saving + "/train/" + "data_clean.csv", index=False)
df_train_idxs = pd.DataFrame(train_idxs, columns=["original_idxs"])
df_train_idxs.to_csv(path_saving + "/train/" + "original_idxs.csv", index=False)

df_validation = df_data.iloc[validation_idxs, :]
df_validation = df_validation.reset_index(drop=True)
df_validation.to_csv(path_saving + "/validation/" + "data_clean.csv", index=False)
df_validation_idxs = pd.DataFrame(validation_idxs, columns=["original_idxs"])
df_validation_idxs.to_csv(path_saving + "/validation/" + "original_idxs.csv", index=False)

name_file = "adult.test"
df_test = pd.read_csv(folder_path + name_file, names=names) # shape 16280
df_test = df_test.drop(columns=['education', 'fnlwgt'])
df_test = df_test.reset_index(drop=True)
df_test.to_csv(path_saving + "/test/" + "data_clean.csv", index=False)
df_test_idxs = pd.DataFrame(np.arange(df_test.shape[0]), columns=["original_idxs"])
df_test_idxs.to_csv(path_saving + "/test/" + "original_idxs.csv", index=False)

# df_full = pd.concat([df_data, df_test])
df_data.to_csv(path_saving + "/full/" + "data_clean.csv", index=False) # TODO: this excludes training






