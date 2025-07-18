import numpy as np



def dgp_choose(dgp_number, m,n,d):
    d = int(d)

    if dgp_number == 1:
        X = np.random.normal(loc=0.0, scale=1.0, size=(m, d))  # Sample from P
        Y = np.random.normal(loc=0.0, scale=1.0, size=(n, d))

    

    elif dgp_number == 2:
        X = np.random.standard_t(df=5, size=(m, d))  # Sample from P
        Y = np.random.standard_t(df=5, size=(n, d))
        
    elif dgp_number == 3:
        X = np.random.chisquare(df=3, size=(m, d))  # Sample from chi-square distribution with df=3
        Y = np.random.chisquare(df=3, size=(n, d))
 
    elif dgp_number == 4:
        X = np.random.poisson(lam=1, size=(m, d))
        Y = np.random.poisson(lam=1, size=(n, d))

    elif dgp_number == 5:
        rho = 0.5
        idx = np.arange(d)
        cov_matrix = rho ** np.abs(idx[:, None] - idx[None, :])

        X = np.random.multivariate_normal(mean=np.zeros(d), cov=cov_matrix, size=m)
        Y = np.random.multivariate_normal(mean=np.zeros(d), cov=cov_matrix, size=n)
  
    
    return Y,X