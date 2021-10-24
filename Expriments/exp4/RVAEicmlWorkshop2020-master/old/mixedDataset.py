
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import reduce
#from torchvision import transforms, utils

from collections import OrderedDict
import abc


class mixedDataset(Dataset):

    def __init__(self, csv_file_path, num_feat_names, cat_feat_names, target_feat_name, is_target_feat_cat):

        self.df_dataset = pd.read_csv(csv_file_path)
        self.dataset_type = 'mixed'

        self.cat_cols = cat_feat_names
        self.num_cols = num_feat_names
        self.target_feat_name = target_feat_name[0] #TODO: assume there is only one target feature for now
        self.is_target_feat_cat = is_target_feat_cat
        self.len_data = self.df_dataset.shape[0]

        ## categorical features
        for col_name in cat_feat_names:
            self.df_dataset[col_name] = self.df_dataset[col_name].astype("category", copy=False)

        if self.is_target_feat_cat:
            self.df_dataset[self.target_feat_name] = self.df_dataset[self.target_feat_name].astype("category", copy=False)
            cats_list = self.df_dataset[self.target_feat_name].cat.categories.tolist()
            self.target_cat_to_idx = OrderedDict([(self.target_feat_name, OrderedDict(zip(cats_list, range(len(cats_list)))))])
            self.target_idx_to_cat = OrderedDict([(self.target_feat_name, OrderedDict(zip(range(len(cats_list)), cats_list)))])
            self.target_cat_name_to_idx = dict([(self.target_feat_name, self.df_dataset.columns.get_loc(self.target_feat_name))])
            self.target_cat_idx_to_name = dict([(self.df_dataset.columns.get_loc(self.target_feat_name), self.target_feat_name)])

        self.cat_to_idx = []
        self.idx_to_cat = []

        for col_name in cat_feat_names:
            cats_list = self.df_dataset[col_name].cat.categories.tolist()
            # from category to code
            self.cat_to_idx.append((col_name, OrderedDict(zip(cats_list, range(len(cats_list))))))

            # from code to category
            self.idx_to_cat.append((col_name, OrderedDict(zip(range(len(cats_list)), cats_list))))

        self.cat_to_idx = OrderedDict(self.cat_to_idx)
        self.idx_to_cat = OrderedDict(self.idx_to_cat)

        self.cat_name_to_idx = dict([(col_name, self.df_dataset.columns.get_loc(col_name)) for col_name in cat_feat_names])
        self.cat_idx_to_name = dict([(self.df_dataset.columns.get_loc(col_name), col_name) for col_name in cat_feat_names])

        ## continuous features
        for col_name in num_feat_names:
            self.df_dataset[col_name] = self.df_dataset[col_name].astype(float, copy=False)

        if not self.is_target_feat_cat:
            # TODO: did not test
            self.df_dataset[self.target_feat_name] = self.df_dataset[self.target_feat_name].astype(float, copy=False)
            self.target_num_name_to_idx = dict([(self.target_feat_name, self.df_dataset.columns.get_loc(self.target_feat_name))])
            self.target_num_idx_to_name = dict([(self.df_dataset.columns.get_loc(self.target_feat_name), self.target_feat_name)])
            # TODO: not sure if we need to standardize target
            self.target_cont_means = self.df_dataset[self.target_feat_name].mean()
            self.target_cont_stds = self.df_dataset[self.target_feat_name].std()

        self.num_name_to_idx = dict([(col_name, self.df_dataset.columns.get_loc(col_name)) for col_name in num_feat_names])
        self.num_idx_to_name = dict([(self.df_dataset.columns.get_loc(col_name), col_name) for col_name in num_feat_names])

        # standardize (re-scale) continuous features defintions
        self.cont_mins = self.df_dataset[self.num_cols].min()
        self.cont_maxs = self.df_dataset[self.num_cols].max()

        # self.cont_means = self.df_dataset[self.num_cols].mean()
        # self.cont_stds = self.df_dataset[self.num_cols].std()

        ## global defs
        if self.cat_cols:
            self.size_tensor_one_hot = reduce((lambda x, y: x + y), map(len, self.cat_to_idx.values())) + len(num_feat_names)
        self.size_tensor_index = len(self.df_dataset.columns)

        #self.bool_cat_array = torch.tensor([1 if (data_col in cat_feat_names) else 0 for data_col in self.df_dataset.columns],
        #                                   dtype=torch.ByteTensor) # if True then categorical

        self.feat_info = []
        for col_name in self.df_dataset.columns:
            if col_name in cat_feat_names: # categorical
                self.feat_info.append((col_name, "categ", len(self.cat_to_idx[col_name])))
            elif col_name in num_feat_names: # numerical (real)
                self.feat_info.append((col_name, "real", 1))
            else:
                if self.is_target_feat_cat:
                    self.feat_info.append((self.target_feat_name, "categ", len(self.target_cat_to_idx[self.target_feat_name])))
                else:
                    self.feat_info.append((self.target_feat_name, "real", 1))


        # get mapping for one-hot representation or expanded
        cursor_feature = 0
        self.start_idx_feature = [[] for col_name in self.df_dataset.columns]

        for col_name in self.df_dataset.columns:
            if col_name in self.cat_cols:
                self.start_idx_feature[self.cat_name_to_idx[col_name]] = cursor_feature
                cursor_feature += len(self.cat_to_idx[col_name])
            elif col_name in self.num_cols:
                self.start_idx_feature[self.num_name_to_idx[col_name]] = cursor_feature
                cursor_feature += 1
            elif col_name==self.target_feat_name:
                if self.is_target_feat_cat:
                    self.start_idx_feature[self.target_cat_name_to_idx[col_name]] = cursor_feature
                    cursor_feature += len(self.target_cat_to_idx[col_name])
                else:
                    self.start_idx_feature[self.target_num_name_to_idx[col_name]] = cursor_feature
                    cursor_feature += 1

    def get_order(self, df_data):

        """ df_data is not one-hot converted yet, assumes same order as self.df_dataset """

        order = []
        for col in df_data.columns:
            if col in self.cat_cols:
                order.extend(['{}_{}'.format(col, categ_name)
                            for categ_name in self.cat_to_idx[col].keys()])
                            # df[col].cat.categories (the same as in defs above)
            elif col in self.num_cols:
                order.append(col)
            elif col == self.target_feat_name:
                if self.is_target_feat_cat:
                    order.extend(['{}_{}'.format(col, categ_name)
                                  for categ_name in self.target_cat_to_idx[col].keys()])
                else:
                    order.append(col)


        return order

    def from_raw_to_one_hot(self, sample_row):

        """ transform each feature value into tensor, then concatenate all in the end """

        ret_tensor = torch.zeros(self.size_tensor_one_hot, dtype=torch.float)
        for col_idx, col_name in enumerate(self.df_dataset.columns):
            if col_name in self.cat_cols:
                idx_cur = self.start_idx_feature[self.cat_name_to_idx[col_name]]
                ret_tensor[idx_cur + self.cat_to_idx[col_name][sample_row[col_idx]]] = 1.
            elif col_name in self.num_cols: # num_cols
                idx_cur = self.start_idx_feature[self.num_name_to_idx[col_name]]
                mean_col = float(self.cont_means[col_name])
                std_col = float(self.cont_stds[col_name])
                ret_tensor[idx_cur] = (sample_row[col_idx] - mean_col) / std_col
            else:
                if self.is_target_feat_cat:
                    idx_cur = self.start_idx_feature[self.target_cat_name_to_idx[col_name]]
                    ret_tensor[idx_cur + self.target_cat_to_idx[col_name][sample_row[col_idx]]] = 1.


    def from_raw_to_index(self, sample_row):
        # TODO: did not modify this
        """ NOTE: categorical codes need to be cast to torch.long in the main code (must cast there)"""

        ret_tensor = torch.zeros(self.size_tensor_index, dtype=torch.float)

        for col_name in self.num_cols:
            mean_col = float(self.cont_means[col_name])
            std_col = float(self.cont_stds[col_name])
            ret_tensor[self.num_name_to_idx[col_name]] = \
            (sample_row[self.num_name_to_idx[col_name]] - mean_col) / std_col

        for col_name in self.cat_cols:
            ret_tensor[self.cat_name_to_idx[col_name]] = \
            self.cat_to_idx[col_name][sample_row[self.cat_name_to_idx[col_name]]]

        return ret_tensor

    def standardize_dataset(self, column, cat_change=True):

        """ NOTE: this method standardize the data for real attributes and get the correct codes """
        """ for categorical attributes."""

        if column.name in self.num_cols:
            column = (column - self.cont_mins[column.name])/(self.cont_maxs[column.name] - self.cont_mins[column.name])
            # column = (column - self.cont_means[column.name])/self.cont_stds[column.name]
        elif cat_change and column.name in self.cat_cols:
            column = column.replace(self.cat_to_idx[column.name])
        elif cat_change and column.name == self.target_feat_name: # TODO: add case when target is numerical
            column = column.replace(self.target_cat_to_idx[column.name])

        return column

    def standardize_dataset_one_hot(self, df_data):
        # TODO: did not modify this
        ret_data = df_data.apply(lambda col: self.standardize_dataset(col, cat_change=False))
        order_cols = self.get_order(df_data)
        ret_data = pd.get_dummies(ret_data, columns=self.cat_cols)[order_cols]

        return ret_data

    def from_index_to_raw(self, sample_row):
        # TODO: did not modify this
        """ NOTE: takes in torch.float tensor, and then casts to int the categorical codes """
        """ used in vae synthetic data gen. """

        ret_tensor = [[] for x in range(self.size_tensor_index)]

        for col_name in self.num_cols:
            mean_col = float(self.cont_means[col_name])
            std_col = float(self.cont_stds[col_name])
            ret_tensor[self.num_name_to_idx[col_name]] = \
            mean_col + sample_row[self.num_name_to_idx[col_name]] * std_col

        for col_name in self.cat_cols:
            ret_tensor[self.cat_name_to_idx[col_name]] = \
            self.idx_to_cat[col_name][int(sample_row[self.cat_name_to_idx[col_name]])]

        return np.array(ret_tensor, dtype=object)

    @abc.abstractmethod
    def __len__(self):
        """Not implemented"""

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Not implemented"""


class mixedDatasetInstance(mixedDataset):

    def __init__(self, csv_file_path_all, csv_file_path_instance,
                 num_feat_names, cat_feat_names, target_feat_name, is_target_feat_cat,
                 anom_frac=None,
                 get_indexes=False,
                 use_one_hot=False):

        super().__init__(csv_file_path_all, num_feat_names, cat_feat_names, target_feat_name, is_target_feat_cat)

        self.df_dataset_instance = pd.read_csv(csv_file_path_instance)

        self.get_indexes = get_indexes
        self.one_hot = use_one_hot

        # make sure of data types in the dataframe
        for col_name in self.cat_cols:
            # makes sure all categories are accounted for (impt. in one-hot enc.)
            cat_type_col = pd.api.types.CategoricalDtype(categories=list(self.cat_to_idx[col_name].keys()))
            self.df_dataset_instance[col_name] = self.df_dataset_instance[col_name].astype(cat_type_col)

        for col_name in self.num_cols:
            self.df_dataset_instance[col_name] = self.df_dataset_instance[col_name].astype(float, copy=False)

        if self.is_target_feat_cat:
            cat_type_col = pd.api.types.CategoricalDtype(categories=list(self.target_cat_to_idx[self.target_feat_name].keys()))
            self.df_dataset_instance[self.target_feat_name] = self.df_dataset_instance[self.target_feat_name].astype(cat_type_col)
        else:
            self.df_dataset_instance[self.target_feat_name] = self.df_dataset_instance[self.target_feat_name].astype(float, copy=False)

        # standardize the dataset here
        self.df_dataset_instance_standardized = self.df_dataset_instance.apply(self.standardize_dataset)

        if anom_frac is not None:
            n_samples = self.df_dataset_instance_standardized.shape[0]
            n_inliers = self.df_dataset_instance_standardized[self.df_dataset_instance_standardized[self.target_feat_name] == 0].shape[0]
            n_outliers = self.df_dataset_instance_standardized[self.df_dataset_instance_standardized[self.target_feat_name] == 1].shape[0] # TODO: we assume outliers are labeled as 1
            ratio_outliers_to_inliers = n_outliers/n_samples
            print("Outlier in training data is %.2f" % ratio_outliers_to_inliers)
            if anom_frac > ratio_outliers_to_inliers:
                raise ValueError("Anomaly fraction cannot be more than the original ratio of outliers to inliers")

            n_picked_outliers = int(n_samples*anom_frac)
            drop_number = int(n_outliers - n_picked_outliers)
            drop_indices = np.random.choice(self.df_dataset_instance_standardized[self.df_dataset_instance_standardized[self.target_feat_name] == 1].index, drop_number, replace=False)
            self.df_dataset_instance_standardized.drop(drop_indices, inplace=True)

    def __len__(self):
        return self.df_dataset_instance_standardized.shape[0]

    def __getitem__(self, idx):
        # if self.get_indexes:
        #     index_ret = [idx]
        # else:
        #     index_ret = []

        # if self.one_hot:
        #     ret_list = [self.from_raw_to_one_hot(self.df_dataset_instance.iloc[idx].values)]
        # else:
        #     ret_list = [self.from_raw_to_index(self.df_dataset_instance.iloc[idx].values)]
        data = self.df_dataset_instance_standardized.drop(self.target_feat_name, axis=1).iloc[idx]
        target = self.df_dataset_instance_standardized.loc[:, self.df_dataset_instance_standardized.columns == self.target_feat_name].iloc[idx]

        data = torch.tensor(data.values.astype(dtype='float32'), dtype=torch.float)
        target = torch.tensor(target.values.astype(dtype='float32'), dtype=torch.float)

        ret_list = (data, target)
        # ret_list += index_ret

        return ret_list


