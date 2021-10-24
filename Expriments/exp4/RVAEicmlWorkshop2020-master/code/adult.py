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

    df = df.apply(lambda x: x.str.strip(' \t.'))

    col_type = [
        ("age", CONTINUOUS),
        ("workclass", CATEGORICAL),
        ("fnlwgt", CONTINUOUS),
        ("education", ORDINAL, ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Prof-school", "Assoc-voc", "Assoc-acdm", "Some-college", "Bachelors", "Masters", "Doctorate"]),
        ("education-num", CONTINUOUS),
        ("marital-status", CATEGORICAL),
        ("occupation", CATEGORICAL),
        ("relationship", CATEGORICAL),
        ("race", CATEGORICAL),
        ("sex", CATEGORICAL),
        ("capital-gain", CONTINUOUS),
        ("capital-loss", CONTINUOUS),
        ("hours-per-week", CONTINUOUS),
        ("native-country", CATEGORICAL),
        ("salary", CATEGORICAL)
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
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain',
             'capital-loss', 'hours-per-week', 'native-country', 'salary']

    target_feat = 'salary'

    df_train = pd.read_csv("./data/adult/adult.data", dtype=str, names=names)
    df_test = pd.read_csv("./data/adult/adult.test", dtype=str, names=names)
    df_train, meta, target_feat_idx = prepare_columns(df_train, target_feat)
    df_test, _, _ = prepare_columns(df_test, target_feat)

    if val_ratio is not None:
        n_samples = df_train.shape[0]
        n_valid_samples = int(n_samples*val_ratio)
        df_train = df_train.iloc[:-n_valid_samples]
        df_val = df_train.iloc[n_valid_samples:]

    if anom_frac is not None:
        n_samples = df_train.shape[0]
        inlier_val = '<=50K'
        outlier_val = '>50K'
        n_inliers = df_train[df_train[target_feat] == inlier_val].shape[0]
        n_outliers = df_train[df_train[target_feat] == outlier_val].shape[0]
        assert n_inliers > n_outliers
        ratio_outliers_to_inliers = n_outliers / n_samples
        print("Outlier in training data is %.2f" % ratio_outliers_to_inliers)
        if anom_frac > ratio_outliers_to_inliers:
            raise ValueError("Anomaly fraction cannot be more than the original fraction of outliers")
        n_picked_outliers = int(n_samples * anom_frac)
        drop_number = int(n_outliers - n_picked_outliers)
        drop_indices = np.random.choice(df_train[df_train[target_feat] == outlier_val].index, drop_number, replace=False)
        df_train.drop(drop_indices, inplace=True)

    t_train = project_table(df_train, meta)
    t_test = project_table(df_test, meta)
    if val_ratio is not None:
        t_val = project_table(df_val, meta)
    else:
        t_val = None

    name = "adult_" + '%.2f' % anom_frac
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, val=t_val, test=t_test, target_feat_idx=target_feat_idx)

    # verify("{}/{}.npz".format(output_dir, name),
    #         "{}/{}.json".format(output_dir, name))

get_train_val_test(0.1, 0)