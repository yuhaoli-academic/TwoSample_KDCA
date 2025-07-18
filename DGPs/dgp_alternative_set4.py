import numpy as np


def generate_covariance_matrix(d, rho=0.5):
    idx = np.arange(d)
    cov_matrix = rho ** np.abs(idx[:, None] - idx[None, :])
    return cov_matrix



def generate_data(d, a, b, m,n):
    cov_matrix = generate_covariance_matrix(d)
    mean_X = np.zeros(d)
    mean_Y = a * np.ones(d)
    cov_Y = b * cov_matrix
    X = 0.8*np.random.multivariate_normal(mean_X, cov_matrix, size=m) + 0.2*np.random.standard_t(df=5, size=(m, d))
    Y = 0.8*np.random.multivariate_normal(mean_Y, cov_Y, size=n) + 0.2*np.random.standard_t(df=3, size=(n, d))
    return X, Y

def dgp_choose_set4(m,n,d,loc,scale):
    X,Y = generate_data(d, loc, scale, m, n)

    return Y,X