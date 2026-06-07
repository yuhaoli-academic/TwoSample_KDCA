import numpy as np

def dgp_choose_set1_2(m,n,d,loc,scale):
    d = int(d)
    frac_d = int(d * 0.1)  # 10% of d

    X = np.random.normal(loc=0.0, scale=1.0, size=(m, d))  # Sample from P
    Y = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
    Y[:, :frac_d] = np.random.normal(loc=loc, scale=scale, size=(n, frac_d))

    return Y,X