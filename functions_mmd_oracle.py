import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import minimize_scalar

# ... [Keep your existing functions] ...

def mmd_u_statistic_and_variance(X, Y, sigma):
    """
    Computes the unbiased MMD^2 U-statistic and its unbiased variance estimate V_m.
    Based on Equation (2) and Equation (5) of Sutherland et al. (2017).
    """
    n = X.shape[0]
    m = Y.shape[0]
    
    # Ensure balanced sample size for this specific variance formula
    if n != m:
        n = min(n, m)
        X = X[:n]
        Y = Y[:n]

    # Compute Kernel Matrices
    K_XX_dists = cdist(X, X, 'sqeuclidean')
    K_YY_dists = cdist(Y, Y, 'sqeuclidean')
    K_XY_dists = cdist(X, Y, 'sqeuclidean')
    
    K_XX = np.exp(-K_XX_dists / sigma)
    K_YY = np.exp(-K_YY_dists / sigma)
    K_XY = np.exp(-K_XY_dists / sigma)
    
    # Zero out diagonals for U-statistic
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    # 1. Compute MMD_U (Equation 2)
    sum_K_XX = np.sum(K_XX)
    sum_K_YY = np.sum(K_YY)
    sum_K_XY = np.sum(K_XY)
    
    mmd_u = (sum_K_XX + sum_K_YY) / (n * (n - 1)) - 2 * sum_K_XY / (n * n)
    
    # 2. Compute Variance Estimate V_m (Equation 5)
    e = np.ones(n)
    
    # Precompute necessary matrix-vector products
    K_XX_e = K_XX @ e
    K_YY_e = K_YY @ e
    K_XY_e = K_XY @ e
    K_XY_T_e = K_XY.T @ e
    
    # Factorials/Pochhammer symbols
    n_2 = n * (n - 1)
    n_3 = n * (n - 1) * (n - 2)
    n_4 = n * (n - 1) * (n - 2) * (n - 3)
    
    # Term A
    term_A = (4 / n_4) * (np.dot(K_XX_e, K_XX_e) + np.dot(K_YY_e, K_YY_e))
    
    # Term B
    term_B_coeff = 4 * (n**2 - n - 1) / (n**3 * (n - 1)**2)
    term_B = term_B_coeff * (np.dot(K_XY_e, K_XY_e) + np.dot(K_XY_T_e, K_XY_T_e))
    
    # Term C
    term_C_denom = n**2 * (n**2 - 3*n + 2)
    term_C_part1 = e.T @ (K_XX @ K_XY) @ e
    term_C_part2 = e.T @ (K_YY @ K_XY.T) @ e
    term_C = -8 / term_C_denom * (term_C_part1 + term_C_part2)
    
    # Term D
    term_D = 8 / (n**2 * n_3) * (np.sum(K_XX) + np.sum(K_YY)) * np.sum(K_XY)
    
    # Term E
    term_E = -2 * (2*n - 3) / (n_2 * n_4) * (np.sum(K_XX)**2 + np.sum(K_YY)**2)
    
    # Term F
    term_F = -4 * (2*n - 3) / (n**3 * (n - 1)**3) * np.sum(K_XY)**2
    
    # Term G
    term_G_denom = n * (n**3 - 6*n**2 + 11*n - 6)
    term_G = -2 / term_G_denom * (np.sum(K_XX**2) + np.sum(K_YY**2))
    
    # Term H
    term_H = 4 * (n - 2) / (n**2 * (n - 1)**3) * np.sum(K_XY**2)
    
    V_m = term_A + term_B + term_C + term_D + term_E + term_F + term_G + term_H
    
    # Safety checks for numerical stability
    if np.isnan(V_m) or np.isinf(V_m) or V_m < 1e-12:
        V_m = 1e-12
        
    return mmd_u, V_m

def optimized_mmd_test(X, Y, n_permutations=1000, seed=None):
    """
    Optimized MMD Two-Sample Test.
    """
    if seed is not None:
        np.random.seed(seed)
        
    N = X.shape[0]
    M = Y.shape[0]
    n_train = N // 2
    m_train = M // 2
    
    # 1. Split Data
    perm_X = np.random.permutation(N)
    perm_Y = np.random.permutation(M)
    
    X_train = X[perm_X[:n_train]]
    X_test = X[perm_X[n_train:]]
    Y_train = Y[perm_Y[:m_train]]
    Y_test = Y[perm_Y[m_train:]]
    
    # 2. Optimize Bandwidth on Training Set
    def objective(log_sigma):
        sigma = np.exp(log_sigma)
        mmd_u, V_m = mmd_u_statistic_and_variance(X_train, Y_train, sigma)
        
        # If values are NaN, return infinity (minimizer avoids this region)
        if np.isnan(mmd_u) or np.isnan(V_m):
            return np.inf
            
        t_stat = mmd_u / np.sqrt(V_m)
        return -t_stat
    
    # Determine search range
    Z_pool = np.vstack((X_train, Y_train))
    dists = pdist(Z_pool, 'euclidean')
    med_dist = np.median(dists)
    
    # Handle edge case where distances are zero
    if med_dist < 1e-6:
        med_dist = 1.0
        
    bounds = (np.log(med_dist / 20), np.log(med_dist * 20))
    
    try:
        res = minimize_scalar(objective, bounds=bounds, method='bounded')
        best_sigma = np.exp(res.x)
    except Exception:
        # Fallback to median heuristic if optimization fails
        best_sigma = med_dist
    
    # 3. Run Test on Test Set
    Z_test = np.vstack((X_test, Y_test))
    n_test = X_test.shape[0]
    m_test = Y_test.shape[0]
    
    dists_test = cdist(Z_test, Z_test, 'sqeuclidean')
    K_test = np.exp(-dists_test / best_sigma)
    
    def compute_mmd_from_K(K, n, m):
        K_XX = K[:n, :n]
        K_YY = K[n:, n:]
        K_XY = K[:n, n:]
        
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        
        mmd = np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY) / (n * m)
        return mmd
    
    obs_mmd = compute_mmd_from_K(K_test, n_test, m_test)
    
    # Permutations
    perm_stats = []
    for _ in range(n_permutations):
        idx = np.random.permutation(n_test + m_test)
        K_perm = K_test[np.ix_(idx, idx)]
        p_mmd = compute_mmd_from_K(K_perm, n_test, m_test)
        perm_stats.append(p_mmd)
        
    perm_stats = np.array(perm_stats)
    
    # 4. Compute P-values
    p_val = (np.sum(perm_stats >= obs_mmd) + 1) / (n_permutations + 1)
        
    return p_val