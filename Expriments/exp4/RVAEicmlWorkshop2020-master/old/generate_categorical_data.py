import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import torch

def get_data(n, q, mu, rho, phi, C=None, enc=None, seed=10):
    np.random.seed(seed)
    ####################################
    mu_0 = np.zeros(q)
    mu_1 = mu*np.ones(q)
    ####################################
    Sigma = rho*np.ones((q, q))
    for i in range(q):
        Sigma[i, i] = 1
    if q == 1:
        Sigma[0][0] = rho

    ####################################
    # if not C:
    #     C = [None]*q
    #     for i in range(int(q/3)):
    #         C[i] = 2
    #     for i in range(int(q/3), int(2*q/3)):
    #         C[i] = 3
    #     for i in range(int(2*q/3), q):
    #         C[i] = 5
    if not C:
        C = [None]*q
        for i in range(int(q/3)):
            C[i] = 10
        for i in range(int(q/3), int(2*q/3)):
            C[i] = 10
        for i in range(int(2*q/3), q):
            C[i] = 10
    assert len(C) == q
    ####################################
    inliers = np.random.multivariate_normal(mu_0, Sigma, int(n*(1-phi)))
    outliers =mu_1 + Sigma * np.random.rand(int(n*phi), q)
    X = np.concatenate([inliers, outliers])

    ####################################
    if not enc:
        enc = KBinsDiscretizer(n_bins=C, encode='onehot', strategy='uniform')
        enc.fit(X)
    X_cat = (enc.transform(X)).toarray()
    return X_cat, enc


def main(n_train, n_test, n_valid,  q, mu_anom, mu_outlier, rho, frac_anom, C, batch_size, seed):
    train_data, enc = get_data(n_train, q, mu_anom, rho, frac_anom, C, None, seed)
    input_dim = train_data.shape[1]

    test_normal_data, _ = get_data(int(n_test/2), q, mu_outlier, rho, 0, C, enc, seed+1)
    test_outlier_data, _ = get_data(int(n_test/2), q, mu_outlier, rho, 1, C, enc, seed+1)

    valid_normal_data, _ = get_data(int(n_valid*0.9), q, mu_outlier, rho, 0, C, enc, seed+2)
    valid_outlier_data, _ = get_data(int(n_valid*0.1), q, mu_outlier, rho, 1, C, enc, seed+2)

    train_loader = torch.utils.data.DataLoader(torch.from_numpy(train_data).float(),
                                               batch_size=batch_size,
                                               shuffle=True)
    test_normal_loader = torch.utils.data.DataLoader(torch.from_numpy(test_normal_data).float(),
                                              batch_size=len(test_normal_data),
                                              shuffle=False)
    test_outlier_loader = torch.utils.data.DataLoader(torch.from_numpy(test_outlier_data).float(),
                                              batch_size=len(test_outlier_data),
                                              shuffle=False)
    valid_normal_loader = torch.utils.data.DataLoader(torch.from_numpy(valid_normal_data).float(),
                                              batch_size=len(valid_normal_data),
                                              shuffle=False)
    valid_outlier_loader = torch.utils.data.DataLoader(torch.from_numpy(valid_outlier_data).float(),
                                              batch_size=len(valid_outlier_data),
                                              shuffle=False)
    return train_loader, test_normal_loader, test_outlier_loader, valid_normal_loader, valid_outlier_loader


if __name__ == '__main__':
    n = 100
    q = 10
    mu = 4
    rho = 0.25
    phi = 0.10
    X_cat = get_data(n, q, mu, rho, phi)
    print(X_cat.shape)


