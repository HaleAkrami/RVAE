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
        "back": "dos",
        "buffer_overflow": "u2r",
        "ftp_write": "r2l",
        "guess_passwd": "r2l",
        "imap": "r2l",
        "ipsweep": "probe",
        "land": "dos",
        "loadmodule": "u2r",
        "multihop": "r2l",
        "neptune": "dos",
        "nmap": "probe",
        "perl": "u2r",
        "phf": "r2l",
        "pod": "dos",
        "portsweep": "probe",
        "rootkit": "u2r",
        "satan": "probe",
        "smurf": "dos",
        "spy": "r2l",
        "teardrop": "dos",
        "warezclient": "r2l",
        "warezmaster": "r2l",
        "normal": "normal"
    }

    df.drop([19], axis=1, inplace=True)

    # df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: label_mapping[x])
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    col_type = [
        ("duration", CONTINUOUS),
        ("protocol_type", CATEGORICAL),
        ("service", CATEGORICAL),
        ("flag", CATEGORICAL),
        ("src_bytes", CONTINUOUS),
        ("dst_bytes", CONTINUOUS),
        ("land", CATEGORICAL),
        ("wrong_fragment", ORDINAL, ['0', '1', '2', '3']),
        ("urgent", ORDINAL, ['0', '1', '2', '3']),
        ("hot", CONTINUOUS),
        ("num_failed_logins", ORDINAL, ['0', '1', '2', '3', '4', '5']),
        ("logged_in", CATEGORICAL),
        ("num_compromised", CONTINUOUS),
        ("root_shell", CATEGORICAL),
        ("su_attempted", ORDINAL, ['0', '1', '2', '3']),
        ("num_root", CONTINUOUS),
        ("num_file_creations", CONTINUOUS),
        ("num_shells", ORDINAL, ['0', '1', '2', '4', '5']),
        ("num_access_files", ORDINAL, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']),
        # ("num_outbound_cmds", CONTINUOUS), # all zero, removed
        ("is_host_login", CATEGORICAL),
        ("is_guest_login", CATEGORICAL),
        ("count", CONTINUOUS),
        ("srv_count", CONTINUOUS),
        ("serror_rate", CONTINUOUS),
        ("srv_serror_rate", CONTINUOUS),
        ("rerror_rate", CONTINUOUS),
        ("srv_rerror_rate", CONTINUOUS),
        ("same_srv_rate", CONTINUOUS),
        ("diff_srv_rate", CONTINUOUS),
        ("srv_diff_host_rate", CONTINUOUS),
        ("dst_host_count", CONTINUOUS),
        ("dst_host_srv_count", CONTINUOUS),
        ("dst_host_same_srv_rate", CONTINUOUS),
        ("dst_host_diff_srv_rate", CONTINUOUS),
        ("dst_host_same_src_port_rate", CONTINUOUS),
        ("dst_host_srv_diff_host_rate", CONTINUOUS),
        ("dst_host_serror_rate", CONTINUOUS),
        ("dst_host_srv_serror_rate", CONTINUOUS),
        ("dst_host_rerror_rate", CONTINUOUS),
        ("dst_host_srv_rerror_rate", CONTINUOUS),
        ("label", CATEGORICAL)
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

    target_feat = 'label'

    df_train = pd.read_csv("./data/kdd/kddcup.data_10_percent", dtype='str', header=None)
    df_test = pd.read_csv("./data/kdd/corrected", dtype='str', header=None)

    # df_train = pd.read_csv("./data/kdd/KDDTrain+.txt", dtype='str', header=None)
    # df_test = pd.read_csv("./data/kdd/KDDTest+.txt", dtype='str', header=None)
    # df_train = df_train[:10000] # TODO: 10K is only for testing, comment this later
    # df_test = df_test[:10000] # TODO: 10K is only for testing, comment this later
    df_all = pd.concat([df_train, df_test]) # test data has some values that are not seen in train, so computing meta based on all data
    _, meta, target_feat_idx = prepare_columns(df_all, target_feat)
    df_train, _, _ = prepare_columns(df_train, target_feat)
    df_test, _, _ = prepare_columns(df_test, target_feat)

    if val_ratio is not None:
        n_samples = df_train.shape[0]
        n_valid_samples = int(n_samples*val_ratio)
        df_train = df_train.iloc[:-n_valid_samples]
        df_val = df_train.iloc[n_valid_samples:]
        # t_train = tdata[:-n_valid_samples]
        # t_val = tdata[-n_valid_samples:]

    if anom_frac is not None:
        n_samples = df_train.shape[0]
        inlier_val = 'normal'
        outlier_val = 'attack'
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
    t_test = project_table(df_test, meta)
    if val_ratio is not None:
        t_val = project_table(df_val, meta)
    else:
        t_val = None

    # np.random.seed(0)
    # np.random.shuffle(tdata)

    name = "original_kdd_" + '%.2f' % anom_frac
    with open("{}/{}.json".format(output_dir, name), 'w') as f:
        json.dump(meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    np.savez("{}/{}.npz".format(output_dir, name), train=t_train, val=t_val, test=t_test, target_feat_idx=target_feat_idx, meta=meta)
    print("saved results")
    # verify("{}/{}.npz".format(output_dir, name),
    #         "{}/{}.json".format(output_dir, name))


if __name__ == '__main__':
    anom_frac_list = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    for anom_frac in anom_frac_list:
        get_train_val_test(0.2, anom_frac)