import json
import os
import urllib
import torch
import numpy as np
import pandas as pd
from transformer_general import GeneralTransformer as DataTransformer
from torch.utils.data import DataLoader, TensorDataset

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

DATA_PATH = "./data/real/"

def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)

    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_dataset(name):
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    return data['train'], data['val'], data['test'], data['target_feat_idx'], categorical_columns, ordinal_columns


def get_loaders(dataset, anom_frac, batch_size):

    if anom_frac is not None:
        dataset = dataset + '_' + '%.2f' % anom_frac
    train, val, test, target_feat_idx, categorical_columns, ordinal_columns = load_dataset(dataset)
    transformer = DataTransformer(target_feat_idx)

    transformer.fit(np.concatenate([train, val, test]), categorical_columns,
                    ordinal_columns)  # TODO: ideally, we should only use training data for transoformation
    train, train_targets = transformer.transform(train)
    val, val_targets = transformer.transform(val)
    test, test_targets = transformer.transform(test)
    data_dim = train.shape[1]
    n_samples = train.shape[0]

    train_dataset = TensorDataset(torch.from_numpy(train.astype('float32')),
                                  torch.from_numpy(train_targets.astype('float32'))
                                  )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)

    test_dataset = TensorDataset(torch.from_numpy(test.astype('float32')),
                                 torch.from_numpy(test_targets.astype('float32'))
                                 )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

    val_dataset = TensorDataset(torch.from_numpy(val.astype('float32')),
                                torch.from_numpy(val_targets.astype('float32'))
                                )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

    return train_loader, test_loader, val_loader, transformer, data_dim, target_feat_idx, categorical_columns, ordinal_columns, n_samples

def generate_synthetic(decoder, n_samples, batch_size, embedding_dim, transformer, meta, target_feat_idx, categorical_columns, ordinal_columns, device):
    decoder.eval()
    steps = n_samples // batch_size + 1
    data = []
    for _ in range(steps):
        mean = torch.zeros(batch_size, embedding_dim)
        std = mean + 1
        noise = torch.normal(mean=mean, std=std).to(device)
        fake, sigmas = decoder(noise)
        fake = torch.tanh(fake)
        data.append(fake.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:n_samples]
    synthetic_data = transformer.inverse_transform(data)

    columns = [meta[id]['name'] for id in range(len(meta)) if id != target_feat_idx]
    df = pd.DataFrame.from_records(synthetic_data)

    def map_to_raw(column, meta):
        if column.name in categorical_columns or column.name in ordinal_columns:
            info = meta[int(column.name)]
            mapper = dict([(id, item) for id, item in enumerate(info['i2s'])])
            column = column.replace(mapper)

        return column

    df_mapped = df.apply(lambda col: map_to_raw(col, meta))
    df_mapped.columns = columns
    return df_mapped


