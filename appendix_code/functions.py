import numpy as np 
from scipy.spatial.distance import cdist,pdist
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal,multivariate_t


def X_gen(m,dim,p):
    """
    m: number of samples
    dim: dimension of the data
    p: distributional choice parameter, p =0 for Gaussian, p=1 for t-distribution
    """
    rho = 0.5
    idx = np.arange(dim)
    cov_matrix = rho ** np.abs(idx[:, None] - idx[None, :])
    if p == 0:
        X = multivariate_normal.rvs(mean=np.zeros(dim), cov=cov_matrix, size=m)
    elif p == 1:
        df = 10
        X = multivariate_t.rvs(loc=np
        .zeros(dim), shape=cov_matrix, df=df, size=m)
    else:
        # Generate i.i.d. Bernoulli random variables with probability p
        choice = np.random.binomial(1, p, size=m)
        gaussian_samples = multivariate_normal.rvs(mean=np.zeros(dim), cov=cov_matrix, size=m)
        other_samples = multivariate_t.rvs(loc=np.zeros(dim), shape=cov_matrix, df=10, size=m)
        X = (1-choice[:, None]) * gaussian_samples + choice[:, None] * other_samples
    return X

def Y_gen(n,dim,p):
    """
    n: number of samples
    dim: dimension of the data
    p: distributional choice parameter, p =0 for Gaussian, p=1 for t-distribution
    """
    rho = 0.5
    idx = np.arange(dim)
    cov_matrix = rho ** np.abs(idx[:, None] - idx[None, :])

    mu1 = 0.1*np.ones(dim)
    if p == 0:
        Y = multivariate_normal.rvs(mean=mu1, cov=1.15*cov_matrix, size=n)
    elif p == 1:
        df = 10
        Y = multivariate_t.rvs(loc=np.zeros(dim), shape=1.22*cov_matrix, df=df, size=n)
    else:
        # Generate i.i.d. Bernoulli random variables with probability p
        choice = np.random.binomial(1, p, size=n)
        gaussian_samples = multivariate_normal.rvs(mean=np.zeros(dim), cov=1.22*cov_matrix, size=n)
        other_samples = multivariate_t.rvs(loc=np.zeros(dim), shape=1.22*cov_matrix, df=10, size=n)
        Y = (1-choice[:, None]) * gaussian_samples + choice[:, None] * other_samples
    return Y


def kernel_maker(l2d_XX, l2d_YY,l2d_XY, sigma,j):
    K_XX = np.exp(-l2d_XX / sigma)
    K_YY = np.exp(-l2d_YY / sigma)
    K_XY = np.exp(-l2d_XY / sigma)
    K_YX = K_XY.T

    K = np.block([[K_XX, K_XY],
                   [K_YX, K_YY]])
    
    eigen_val,eigen_vec = eigsh(K, k=j, which='LM')

    K_j = eigen_vec @ np.diag(eigen_val) @ eigen_vec.T
    return K_j


def multi(mm,nn,d,p,Nrep,Nb,j):
    def process_ss(ss):
        X = X_gen(mm, d, p)
        Y = Y_gen(nn, d, p)
        m = X.shape[0]
        n = Y.shape[0]
        Z = np.vstack((X, Y))
        

        pairwise_dists = pdist(Z, 'euclidean')**2
        sigma = np.median(pairwise_dists)
        
        sigma_candidates = [s * sigma for s in [0.5, 1.0, 2.0, 1/np.sqrt(2.0)]]

        l2d_XX = cdist(X, X, metric='euclidean') ** 2
        l2d_YY = cdist(Y, Y, metric='euclidean') ** 2
        l2d_XY = cdist(X, Y, metric='euclidean') ** 2

        K_j = np.zeros((m+n, m+n))
        for sigma in sigma_candidates:
            K_j += kernel_maker(l2d_XX, l2d_YY, l2d_XY, sigma,j)

        # Select K_XX from K_j
        K_XX_from_K_j = K_j[:m, :m]
            
        # Select K_YY from K_j
        K_YY_from_K_j = K_j[m:, m:]
        
        # Select K_XY from K_j
        K_XY_from_K_j = K_j[:m, m:]
        

        k_X_non_diag = K_XX_from_K_j[np.triu_indices_from(K_XX_from_K_j, k=1)]

        k_Y_non_diag = K_YY_from_K_j[np.triu_indices_from(K_YY_from_K_j, k=1)]
        
        k_XY_flat = K_XY_from_K_j.flatten()

        stat_ker =(m+n)*( np.mean(k_X_non_diag) + np.mean(k_Y_non_diag) - 2 * np.mean(k_XY_flat)) 

        # Bootstrap loop
        if m>=n:
            # Center K_XX
            C = np.eye(m) - np.ones((m, m)) / m
            K_centered = C @ K_XX_from_K_j @C
            p_hat = m / (m+n)
            K_centered = K_centered / m
        else:
            # Center K_YY
            C = np.eye(n) - np.ones((n, n)) / n
            K_centered = C @ K_YY_from_K_j @ C
            p_hat = n / (m+n)
            K_centered = K_centered / n
        
        mu = np.zeros(K_centered.shape[0])
        Sigma = (1 / (p_hat * (1 - p_hat))) * np.eye(K_centered.shape[0])

        v_mat = multivariate_normal.rvs(mean=mu, cov=Sigma, size=Nb)

        stat_kerb=np.sum((v_mat.T * (K_centered @ v_mat.T)),axis=0) - (1 / (p_hat * (1 - p_hat))) * np.trace(K_centered)

        
        # P-value computation
        pvalue_ker = np.mean(stat_ker < stat_kerb)
        
        return (pvalue_ker < 0.1, pvalue_ker < 0.05, pvalue_ker < 0.01)
    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))
    
    # Unpack results
    rej_90, rej_95, rej_99 = zip(*results)
    
    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()


def single(mm,nn,d,p,Nrep,Nb,j):
    def process_ss(ss):
        X = X_gen(mm, d, p)
        Y = Y_gen(nn, d, p)
        m = X.shape[0]
        n = Y.shape[0]
        Z = np.vstack((X, Y))
        

        pairwise_dists = pdist(Z, 'euclidean')**2
        sigma = np.median(pairwise_dists)

        l2d_XX = cdist(X, X, metric='euclidean') ** 2
        l2d_YY = cdist(Y, Y, metric='euclidean') ** 2
        l2d_XY = cdist(X, Y, metric='euclidean') ** 2

        
        K_j = kernel_maker(l2d_XX, l2d_YY, l2d_XY,sigma,j)

        # Select K_XX from K_j
        K_XX_from_K_j = K_j[:m, :m]
            
        # Select K_YY from K_j
        K_YY_from_K_j = K_j[m:, m:]
        
        # Select K_XY from K_j
        K_XY_from_K_j = K_j[:m, m:]
        

        k_X_non_diag = K_XX_from_K_j[np.triu_indices_from(K_XX_from_K_j, k=1)]

        k_Y_non_diag = K_YY_from_K_j[np.triu_indices_from(K_YY_from_K_j, k=1)]
        
        k_XY_flat = K_XY_from_K_j.flatten()

        stat_ker =(m+n)*( np.mean(k_X_non_diag) + np.mean(k_Y_non_diag) - 2 * np.mean(k_XY_flat)) 

        # Bootstrap loop
        if m>=n:
            # Center K_XX
            C = np.eye(m) - np.ones((m, m)) / m
            K_centered = C @ K_XX_from_K_j @C
            p_hat = m / (m+n)
            K_centered = K_centered / m
        else:
            # Center K_YY
            C = np.eye(n) - np.ones((n, n)) / n
            K_centered = C @ K_YY_from_K_j @ C
            p_hat = n / (m+n)
            K_centered = K_centered / n
        
        mu = np.zeros(K_centered.shape[0])
        Sigma = (1 / (p_hat * (1 - p_hat))) * np.eye(K_centered.shape[0])

        v_mat = multivariate_normal.rvs(mean=mu, cov=Sigma, size=Nb)

        stat_kerb=np.sum((v_mat.T * (K_centered @ v_mat.T)),axis=0) - (1 / (p_hat * (1 - p_hat))) * np.trace(K_centered)

        
        # P-value computation
        pvalue_ker = np.mean(stat_ker < stat_kerb)
        
        return (pvalue_ker < 0.1, pvalue_ker < 0.05, pvalue_ker < 0.01)
    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))
    
    # Unpack results
    rej_90, rej_95, rej_99 = zip(*results)
    
    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()


def mmd(mm,nn,d,p,Nrep,Nb):
    def process_ss(ss):
        X = X_gen(mm, d, p)
        Y = Y_gen(nn, d, p)
        m = X.shape[0]
        n = Y.shape[0]
        Z = np.vstack((X, Y))
        

        pairwise_dists = pdist(Z, 'euclidean')**2
        sigma = np.median(pairwise_dists)

        l2d_XX = cdist(X, X, metric='euclidean') ** 2
        l2d_YY = cdist(Y, Y, metric='euclidean') ** 2
        l2d_XY = cdist(X, Y, metric='euclidean') ** 2
        K_XX = np.exp(-l2d_XX / sigma)
        K_YY = np.exp(-l2d_YY / sigma)
        K_XY = np.exp(-l2d_XY / sigma)
        
        k_X_non_diag = K_XX[np.triu_indices_from(K_XX, k=1)]
        k_Y_non_diag = K_YY[np.triu_indices_from(K_YY, k=1)]
        k_XY_flat = K_XY.flatten()

        
        
        # Main statistic
        stat_ker =(m+n)*( np.mean(k_X_non_diag) + np.mean(k_Y_non_diag) - 2 * np.mean(k_XY_flat)) 
        
        stat_kerb = np.zeros(Nb)
        for b in range(Nb):
            idx = np.random.permutation(m + n)
            Xb = Z[idx[:m], :]
            Yb = Z[idx[m:], :]

            K_XXb = np.exp(-cdist(Xb, Xb, metric='euclidean') ** 2 / sigma)
            K_YYb = np.exp(-cdist(Yb, Yb, metric='euclidean') ** 2 / sigma)
            K_XYb = np.exp(-cdist(Xb, Yb, metric='euclidean') ** 2 / sigma)

            k_X_non_diag_b = K_XXb[np.triu_indices_from(K_XXb, k=1)]
            k_Y_non_diag_b = K_YYb[np.triu_indices_from(K_YYb, k=1)]
            k_XY_flat_b = K_XYb.flatten()

            

            stat_kerb[b] = (m + n) * ( np.mean(k_X_non_diag_b) + np.mean(k_Y_non_diag_b) - 2 * np.mean(k_XY_flat_b)) 
        
        # P-value computation
        pvalue_ker = np.mean(stat_ker < stat_kerb)

        return (pvalue_ker < 0.1, pvalue_ker < 0.05, pvalue_ker < 0.01)
    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))
    
    # Unpack results
    rej_90, rej_95, rej_99 = zip(*results)
    
    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()


    

        