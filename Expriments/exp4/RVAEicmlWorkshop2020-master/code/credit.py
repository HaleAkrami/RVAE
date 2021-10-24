# Generate adult datasets

import os
import logging
import json
import numpy as np
import pandas as pd

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

output_dir = "data/real/"
temp_dir = "tmp/"


def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')

    for id_, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, id_] = data.iloc[:, id_].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
            values[:, id_] = mapped
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
    return values


def prepare_columns(df, target_feat):
    """
    returns train and validation
    Args:
        val_ratio:
        frac_anom:

    Returns:

    """
    try:
        os.mkdir(output_dir)
    except:
        pass

    try:
        os.mkdir(temp_dir)
    except:
        pass

    # df = df.apply(lambda x: x.str.strip(' \t.'))

    col_type = [
        ("LIMIT_BAL", CONTINUOUS),
        ("SEX", CATEGORICAL),
        ("EDUCATION", CATEGORICAL),
        ("MARRIAGE", CATEGORICAL),
        ("AGE", CONTINUOUS),
        ("PAY_0", CATEGORICAL),
        ("PAY_2", CATEGORICAL),
        ("PAY_3", CATEGORICAL),
        ("PAY_4", CATEGORICAL),
        ("PAY_5", CATEGORICAL),
        ("PAY_6", CATEGORICAL),
        ("BILL_AMT1", CONTINUOUS),
        ("BILL_AMT2", CONTINUOUS),
        ("BILL_AMT3", CONTINUOUS),
        ("BILL_AMT4", CONTINUOUS),
        ("BILL_AMT5", CONTINUOUS),
        ("BILL_AMT6", CONTINUOUS),
        ("PAY_AMT1", CONTINUOUS),
        ("PAY_AMT2", CONTINUOUS),
        ("PAY_AMT3", CONTINUOUS),
        ("PAY_AMT4", CONTINUOUS),
        ("PAY_AMT5", CONTINUOUS),
        ("PAY_AMT6", CONTINUOUS),
        ("default payment next month", CATEGORICAL)

    ]

    meta = []
    for id_, info in enumerate(col_type):
        if info[0] == target_feat:
            # keep track of target feat index so we exclude from training
            target_feat_idx = id_

        if info[1] == CONTINUOUS:
            meta.append({
                "name": info[0],
                "type": info[1],
                "min": np.min(df.iloc[:, id_].values.astype('float')),
                "max": np.max(df.iloc[:, id_].values.astype('float'))
            })
        else:
            if info[1] == CATEGORICAL:
                value_count = list(dict(df.iloc[:, id_].value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
            else:
                mapper = info[2]

            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })
    return df, meta, target_feat_idx


def get_train_val_test(val_ratio=None, anom_frac=None):
    """
    Returns train, validation and test data in npz format
    Args:
        val_ratio: ratio of validation set in train set
        anom_frac: fraction of anomalies
    Returns:

    """
    # names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    #          'occupation', 'relationship', 'race', 'sex', 'capital-gain',
    #          'capital-loss', 'hours-per-week', 'native-country', 'salary']

    target_feat = 'default payment next month'

    df_train = pd.read_csv("./data/DefaultCredit/DefaultCredit/train/data_clean.csv")
    df_val = pd.read_csv("./data/DefaultCredit/DefaultCredit/validation/data_clean.csv")
    df_test = pd.read_csv("./data/DefaultCredit/DefaultCredit/test/data_clean.csv")

    df_all = pd.concat([df_train, df_val, df_test]) # test data has some values that are not seen in train, so computing meta based on all data
    _, meta, target_feat_idx = prepare_columns(df_all, target_feat)
    df_test, _, _ = prepare_columns(df_test, target_feat)
    df_train, _, _ = prepare_columns(df_train, target_feat)
    df_val, _, _ = prepare_columns(df_val, target_feat)

    if anom_frac is not None:
        n_samples = df_train.shape[0]
        inlier_val = 0
        outlier_val = 1
        n_inliers = df_train[df_train.iloc[:, target_feat_idx] == inlier_val].shape[0]
        n_outliers = df_train[df_train.iloc[:, target_feat_idx] == outlier_val].shape[0]
        # assert n_inliers > n_outliers # disable this for kdd data because it has more outliers in training data
        ratio_outliers_to_inliers = n_outliers / n_samples
        print("Outlier in training data is %.2f" % ratio_outliers_to_inliers)
        if anom_frac > ratio_outliers_to_inliers:
            raise ValueError("Anomaly fraction cannot be more than the original fraction of outliers")
        n_picked_outliers = int(n_samples * anom_frac)
        drop_number = int(n_outliers - n_picked_outliers)
        drop_indices = np.random.choice(df_train[df_train.iloc[:, target_feat_idx]  == outlier_val].index, drop_number, replace=False)
        df_train.drop(drop_indices, inplace=True)

    t_train = project_table(df_train, meta)
    t_val = project_table(df_val, meta)
    t_test = project_table(df_test, meta)
    # if val_ratio is not None:
    #     t_val = project_table(df_val, meta)
    # else:
    #     t_val = None

    name = "credit_" + '%.2f' % anom_frac
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, val=t_val, test=t_test, target_feat_idx=target_feat_idx, meta=meta)
    print("saved results")

    # verify("{}/{}.npz".format(output_dir, name),
    #         "{}/{}.json".format(output_dir, name))

if __name__ == '__main__':
    anom_frac_list = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    for anom_frac in anom_frac_list:
        get_train_val_test(None, anom_frac)