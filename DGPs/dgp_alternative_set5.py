import numpy as np


def generate_covariance_matrix(d, rho=0.5):
    idx = np.arange(d)
    cov_matrix = rho ** np.abs(idx[:, None] - idx[None, :])
    return cov_matrix



def dgp_choose_set5(m,n,d,loc,scale):
    cov_matrix = generate_covariance_matrix(d)
    u1 = np.random.chisquare(df=3, size=(m, d)) 
    u2 = np.random.chisquare(df=3, size=(n, d)) 

    X = (cov_matrix @ u1.T).T
    Y = (scale * cov_matrix @ u2.T).T + loc * np.ones((n, d))

    return Y,X