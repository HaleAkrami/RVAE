import pandas as pd
import numpy as np
import os
import errno
import json
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy
from utils import create_data_folders, create_data_splits, save_datasets



# 1 - Select dataset
dataset = 'DefaultCredit' #'Wine' # 'Letter'
name_file = "DefaultCredit.csv"
folder_path = "./DefaultCredit/"

df_data = pd.read_csv(folder_path + name_file)

# real features
# num_feat_names = []
num_feat_names = ["LIMIT_BAL",
                  "AGE",
                  "BILL_AMT1",
                  "BILL_AMT2",
                  "BILL_AMT3",
                  "BILL_AMT4",
                  "BILL_AMT5",
                  "BILL_AMT6",
                  "PAY_AMT1",
                  "PAY_AMT2",
                  "PAY_AMT3",
                  "PAY_AMT4",
                  "PAY_AMT5",
                  "PAY_AMT6"]

## if noise + data should be rounded to next integer value
##      Note: follows the same order as num_feat_names list!
# int_cast_array = np.array([0,0,0,0,0,0,0,0,0,0,0,1], dtype=bool)
int_cast_array = np.ones(14, dtype=bool)

# categorical features
# cat_feat_names = ["SEX"]

cat_feat_names = ["SEX",
                  "EDUCATION",
                  "MARRIAGE",
                  "PAY_0",
                  "PAY_2",
                  "PAY_3",
                  "PAY_4",
                  "PAY_5",
                  "PAY_6"]

target_feat_name = ["default payment next month"]
is_target_feat_cat = True
train_size = 0.8 # TODO: change this ratio for real training
valid_size = 0.1
test_size = 0.1

run_stats = {"name":dataset, "train_size": train_size, "valid_size": valid_size, "test_size": test_size,
             "cat_cols_names":cat_feat_names, "num_cols_names":num_feat_names, "target_feat_name":target_feat_name,
             "is_target_feat_cat": is_target_feat_cat}

cols_info = {"cat_cols_names": run_stats["cat_cols_names"],
             "num_cols_names": run_stats["num_cols_names"],
             "target_feat_name": run_stats["target_feat_name"],
             "is_target_feat_cat": run_stats["is_target_feat_cat"],
             "dataset_type": "mixed"
             }

## create folders
path_saving = create_data_folders(run_stats, folder_path)

with open(path_saving + 'cols_info.json', 'w') as outfile:
    json.dump(cols_info, outfile, indent=4, sort_keys=True)

## get dataset splits (Train; Valid; Test) and entire dataset
train_idxs, validation_idxs, test_idxs = create_data_splits(run_stats, df_data)

## create dataset splits and save to folders
save_datasets(path_saving, df_data, train_idxs, validation_idxs, test_idxs)


