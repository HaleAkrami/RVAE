import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OneHotEncoder
# from sklearn.utils._testing import ignore_warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.

    Args:
        n_cluster (int):
            Number of modes.
        epsilon (float):
            Epsilon value.
    """

    def __init__(self, target_feat_idx, n_clusters=10, epsilon=0.005):
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.target_feat_idx = target_feat_idx

    # @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, column, data):
        return {
            'model': 'simple',
            'mean': data.mean(),
            'std': data.std(),
            'name': column,
            'output_info': [(1, 'tanh')],
            'output_dimensions': 1
        }

    def _fit_discrete(self, column, data):
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(data)
        categories = len(ohe.categories_[0])

        return {
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    def fit(self, data, discrete_columns=tuple()):
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        self.dtypes = data.infer_objects().dtypes
        self.meta = []
        for column in data.columns:
            column_data = data[[column]].values
            if column in discrete_columns:
                meta = self._fit_discrete(column, column_data)
            else:
                meta = self._fit_continuous(column, column_data)
            self.meta.append(meta)
            if column != self.target_feat_idx:
                self.output_info += meta['output_info']
                self.output_dimensions += meta['output_dimensions']


    def _transform_continuous(self, column_meta, data):
        means = column_meta['mean']
        stds = column_meta['std']
        features = (data - means) / (4 * stds)

        return features

    def _transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        return encoder.transform(data)

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        targets = []
        for meta in self.meta:
            column_data = data[[meta['name']]].values
            if 'model' in meta and meta['name'] != self.target_feat_idx:
                values.append(self._transform_continuous(meta, column_data))
            elif 'model' not in meta and meta['name'] != self.target_feat_idx:
                values.append(self._transform_discrete(meta, column_data))
            elif meta['name'] == self.target_feat_idx:
                # assuming target feature is categorical for now
                targets = column_data

        return np.concatenate(values, axis=1).astype(float), np.array(targets)

    def _inverse_transform_continuous(self, meta, data, sigma):
        # TODO: fix this for simple case
        model = meta['model']
        components = meta['components']

        u = data[:, 0]
        v = data[:, 1:]

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), self.n_clusters)) * -100
        v_t[:, components] = v
        v = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, meta, data):
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data, sigmas):
        # TODO: this will fail if target is not in the last column
        start = 0
        output = []
        column_names = []
        for meta in self.meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                sigma = sigmas[start] if sigmas else None
                inverted = self._inverse_transform_continuous(meta, columns_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names).astype(self.dtypes)
        if not self.dataframe:
            output = output.values

        return output


