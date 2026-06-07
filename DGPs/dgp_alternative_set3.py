import numpy as np

def dgp_choose_set3(m,n,d,df):
    d = int(d)
    frac_d = int(d * 0.1)  # 10% of d

    X1 = np.random.normal(loc=0.0, scale=1.0, size=(m, (d - frac_d)))  # Sample from P
    X2 = np.random.standard_t(df=df, size=(m, frac_d))
    X = np.concatenate([X1, X2], axis=1)
    Y = np.random.normal(loc=0.0, scale=1.0, size=(n, d))

    return Y, X