import numpy as np
from scipy.linalg import toeplitz, cholesky

def generate_covariance_matrix(d, rho=0.5):
    """Fast Toeplitz covariance using scipy.linalg.toeplitz"""
    if rho == 0:
        return np.eye(d)
    if rho == 1:
        return np.ones((d, d))
    first_row = rho ** np.arange(d)
    return toeplitz(first_row)

def fast_mvn(rng, mean, cov, size):
    """
    Faster multivariate normal using pre‑computed Cholesky.
    Assumes `cov` is positive definite.
    """
    L = cholesky(cov, lower=True)           # compute once per covariance matrix
    x = rng.normal(size=(size, len(mean)))  # standard normals
    return mean + x @ L.T                   # transform

def generate_data(d, a, b, m, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()       # modern, faster RNG
    
    cov_matrix = generate_covariance_matrix(d)
    mean_X = np.zeros(d)
    mean_Y = a * np.ones(d)
    cov_Y = b * cov_matrix
    
    # X = 0.8 * N(0, cov) + 0.2 * t(5)
    X_normal = fast_mvn(rng, mean_X, cov_matrix, m)
    X_t = rng.standard_t(df=5, size=(m, d))
    X = 0.8 * X_normal + 0.2 * X_t
    
    # Y = 0.8 * N(a·1, b·cov) + 0.2 * t(3)
    Y_normal = fast_mvn(rng, mean_Y, cov_Y, n)
    Y_t = rng.standard_t(df=3, size=(n, d))
    Y = 0.8 * Y_normal + 0.2 * Y_t
    
    return X, Y

def dgp_choose_set4(m, n, d, loc, scale):
    X, Y = generate_data(d, loc, scale, m, n)
    return Y, X   # keep original swapped order