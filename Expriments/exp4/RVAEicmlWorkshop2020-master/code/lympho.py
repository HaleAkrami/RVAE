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

    label_mapping = {
        "1": "outlier",
        "4": "outlier",
        "2": "inlier",
        "3": "inlier",
    }

    # df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: label_mapping[x])
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: label_mapping[x])


    col_type = [
        ("class", CATEGORICAL),
        ("lymphatics", CATEGORICAL),
        ("block of affere", CATEGORICAL),
        ("bl. of lymph. c", CATEGORICAL),
        ('bl. of lymph. s', CATEGORICAL),
        ('by pass', CATEGORICAL),
        ('extravasates', CATEGORICAL),
        ('regeneration of', CATEGORICAL),
        ('early uptake in', CATEGORICAL),
        ('lym.nodes dimin', ORDINAL, ['0', '1', '2', '3']),
        ('lym.nodes enlar', ORDINAL, ['1', '2', '3', '4']),
        ( 'changes in lym.', CATEGORICAL),
        ( 'defect in node', CATEGORICAL),
        ( 'changes in node', CATEGORICAL),
        ( 'changes in stru', CATEGORICAL),
        ( 'special forms', CATEGORICAL),
        ( 'dislocation of', CATEGORICAL),
        ( 'exclusion of no', CATEGORICAL),
        ('no. of nodes in', ORDINAL, ['1', '2', '3', '4', '5', '6', '7', '8'])
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

    target_feat = 'class'

    # df_train = pd.read_csv("./data/kdd/kddcup.data_10_percent", dtype='str', header=None)
    # df_test = pd.read_csv("./data/kdd/corrected", dtype='str', header=None)

    names = ['class', 'lymphatics', 'block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass',
                 'extravasates', 'regeneration of', 'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar',
                 'changes in lym.', 'defect in node', 'changes in node', 'changes in stru', 'special forms',
                 'dislocation of', 'exclusion of no', 'no. of nodes in']

    df_train = pd.read_csv("./data/lymphography/lymphography.data", dtype=str, names=names)
    # df_train = df_train[:10000] # TODO: 10K is only for testing, comment this later
    # df_test = df_test[:10000] # TODO: 10K is only for testing, comment this later
    # df_all = pd.concat([df_train, df_test]) # test data has some values that are not seen in train, so computing meta based on all data
    df_train, meta, target_feat_idx = prepare_columns(df_train, target_feat)
    # df_train, _, _ = prepare_columns(df_train, target_feat)
    # df_test, _, _ = prepare_columns(df_test, target_feat)

    # if val_ratio is not None:
    #     n_samples = df_train.shape[0]
    #     n_valid_samples = int(n_samples*val_ratio)
    #     df_train = df_train.iloc[:-n_valid_samples]
    #     df_val = df_train.iloc[n_valid_samples:]
    #     # t_train = tdata[:-n_valid_samples]
    #     # t_val = tdata[-n_valid_samples:]
    #
    # if anom_frac is not None:
    #     n_samples = df_train.shape[0]
    #     inlier_val = 'normal'
    #     outlier_val = 'attack'
    #     n_inliers = df_train[df_train.iloc[:, target_feat_idx] == inlier_val].shape[0]
    #     n_outliers = df_train[df_train.iloc[:, target_feat_idx] == outlier_val].shape[0]
    #     # assert n_inliers > n_outliers # disable this for kdd data because it has more outliers in training data
    #     ratio_outliers_to_inliers = n_outliers / n_samples
    #     print("Outlier in training data is %.2f" % ratio_outliers_to_inliers)
    #     if anom_frac > ratio_outliers_to_inliers:
    #         raise ValueError("Anomaly fraction cannot be more than the original fraction of outliers")
    #     n_picked_outliers = int(n_samples * anom_frac)
    #     drop_number = int(n_outliers - n_picked_outliers)
    #     drop_indices = np.random.choice(df_train[df_train.iloc[:, target_feat_idx]  == outlier_val].index, drop_number, replace=False)
    #     df_train.drop(drop_indices, inplace=True)

    tdata = project_table(df_train, meta)
    # t_test = project_table(df_test, meta)
    # if val_ratio is not None:
    #     t_val = project_table(df_val, meta)
    # else:
    #     t_val = None

    np.random.seed(10)
    np.random.shuffle(tdata)

    n_samples = tdata.shape[0]
    n_test = int(n_samples*0.3)
    n_val = int(n_samples*0.2)

    t_train = tdata[:-(n_test+n_val)]
    t_val = tdata[-(n_test+n_val):-n_test]
    t_test = tdata[-n_test:]

    print("Train: no of inliers is %d, no of outliers is %d" %(t_train[t_train[:, 0]==0].shape[0], t_train[t_train[:, 0]==1].shape[0]))
    print("Val: no of inliers is %d, no of outliers is %d" % ( t_val[t_val[:, 0] == 0].shape[0], t_val[t_val[:, 0] == 1].shape[0]))
    print("Test: no of inliers is %d, no of outliers is %d" % ( t_test[t_test[:, 0] == 0].shape[0], t_test[t_test[:, 0] == 1].shape[0]))


    # since the data is too small. we use only taining data
    name = "lympho" # + '%.2f' % anom_frac
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, val=t_val, test=t_test, target_feat_idx=target_feat_idx, meta=meta)
    print("saved results")
    # verify("{}/{}.npz".format(output_dir, name),
    #         "{}/{}.json".format(output_dir, name))


if __name__ == '__main__':
    get_train_val_test(None, None)