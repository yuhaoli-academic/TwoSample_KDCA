"""
Spectral Regularized Kernel Two-Sample Tests
Omar Hagrass, Bharath K. Sriperumbudur, Bing Li
"""


import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh, sqrtm

def compute_spectral_statistic_grid(X_test, Y_test, Z, lambdas, h_base, h_mults):
    """
    Computes statistics for a grid of lambdas and bandwidth multipliers.
    Matches the R implementation logic exactly.
    """
    n = X_test.shape[0]
    m = Y_test.shape[0]
    s = Z.shape[0]
    
    # 1. Eigen-decomposition of Covariance Operator on Z (Computed once)
    dist_Z = cdist(Z, Z, 'sqeuclidean')
    
    # Centering matrix H_tilde
    H = np.eye(s) - 1.0 / s * np.ones((s, s))
    H_tilde = (s / (s - 1)) * H
    H1 = sqrtm(H_tilde).real
    
    # 2. Cross distances (Computed once)
    dist_XX = cdist(X_test, X_test, 'sqeuclidean')
    dist_YY = cdist(Y_test, Y_test, 'sqeuclidean')
    dist_XZ = cdist(X_test, Z, 'sqeuclidean')
    dist_YZ = cdist(Y_test, Z, 'sqeuclidean')
    dist_XY = cdist(X_test, Y_test, 'sqeuclidean')
    
    results = []
    
    # Iterate over bandwidth multipliers (h_mults)
    for h_mult in h_mults:
        # R: h_mult is effectively bandwidth multiplier. 
        # R code: K_s <- Ks^(1/h_mult). 
        # Ks = exp(-0.5 * d^2 / h_base).
        # K_s = exp(-0.5 * d^2 / (h_base * h_mult)).
        # So effective bandwidth = h_base * h_mult.
        
        bw = h_base * h_mult
        
        # Kernel on Z
        K_Z = np.exp(-0.5 * dist_Z / bw)
        
        # Eigen decomposition
        Mat = H1 @ K_Z @ H1
        eig_vals, eig_vecs = eigh(Mat)
        
        # Sort descending
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        # Filter eigenvalues
        pos_idx = eig_vals > 1e-9
        eig_vals = eig_vals[pos_idx]
        eig_vecs = eig_vecs[:, pos_idx]
        eig_vals = eig_vals / s
        
        # Kernels for X, Y
        K_n = np.exp(-0.5 * dist_XX / bw)
        K_m = np.exp(-0.5 * dist_YY / bw)
        K_ns = np.exp(-0.5 * dist_XZ / bw)
        K_ms = np.exp(-0.5 * dist_YZ / bw)
        K_mn = np.exp(-0.5 * dist_XY / bw)
        
        # Iterate over Lambdas
        for ld in lambdas:
            # Tikhonov Regularizer: g(x) = 1/(x+ld)
            g0 = 1.0 / ld
            coeff = -1.0 / (ld * (eig_vals + ld))
            G = eig_vecs @ np.diag(coeff) @ eig_vecs.T
            
            term1_part = K_ns @ H1 @ G @ H1 @ K_ns.T
            mat1 = g0 * K_n + (1.0/s) * term1_part
            
            term2_part = K_ms @ H1 @ G @ H1 @ K_ms.T
            mat2 = g0 * K_m + (1.0/s) * term2_part
            
            term3_part = K_ms @ H1 @ G @ H1 @ K_ns.T
            inner_mat3 = g0 * K_mn + (1.0/s) * term3_part
            mat3 = np.sum(inner_mat3)
            
            tr1 = np.trace(mat1)
            sum1 = np.sum(mat1)
            tr2 = np.trace(mat2)
            sum2 = np.sum(mat2)
            
            stat = (sum1 - tr1) / (n * (n - 1)) + \
                   (sum2 - tr2) / (m * (m - 1)) - \
                   2.0 * mat3 / (n * m)
                   
            results.append(stat)
            
    return np.array(results)

def spectral_regularized_test(X, Y, alpha=0.05, s=None, n_permutations=500):
    """
    Performs the Spectral Regularized Kernel Two-Sample Test with Grid Search.
    """
    N, d = X.shape
    M, _ = Y.shape
    
    # --- safe sample splitting ---
    if s is None:
        s = min(N, M) // 2
    else:
        s = min(s, min(N, M) // 2)   # ensure s <= min(N, M)
    if s < 2:
        raise ValueError("s must be at least 2")
    
    # 1. Split Data
    # R code logic: Z is constructed from first s samples. Test is on remaining.
    # We simulate 'first s' by taking random permutation
    perm_X = np.random.permutation(N)
    perm_Y = np.random.permutation(M)
    
    X_1 = X[perm_X[:s]]
    X_test = X[perm_X[s:]]
    
    Y_1 = Y[perm_Y[:s]]
    Y_test = Y[perm_Y[s:]]
    
    n_test = X_test.shape[0]
    m_test = Y_test.shape[0]
    
    if n_test < 2 or m_test < 2:
        return 1.0
        
    # Construct Z (Bernoulli selection)
    mask = np.random.randint(0, 2, s).astype(bool)
    Z = np.where(mask[:, None], X_1, Y_1)
    
    # 2. Parameters from R Script
    # Lambda grid
    # R: Lambda <- 10^seq(-6, 1, 0.75)
    exps = np.arange(-6, 1.01, 0.75)
    lambdas = 10**exps
    
    # Bandwidth Multipliers
    # R: h_mult_arr <- 10^seq(-h_low, h_up, 0.5)
    # Default in R script args: h_low=2, h_up=2 -> 10^0 = 1. So just [1].
    # But let's allow the grid as per the 'sim_iter' logic defaults if needed.
    # Using defaults from R script provided:
    h_low = 2
    h_up = 2
    h_mults = 10**np.arange(-h_low, h_up + 0.01, 0.5)
    
    # Base Bandwidth h
    # R: agg_dist <- pdist(t(agg_samples)); h <- median(agg_dist)^2
    Z_pool = np.vstack((X, Y))
    dists = pdist(Z_pool, 'euclidean')
    h_base = np.median(dists)**2
    
    # 3. Observed Statistics (Vector of length K)
    obs_stats = compute_spectral_statistic_grid(X_test, Y_test, Z, lambdas, h_base, h_mults)
    K = len(obs_stats)
    
    # 4. Permutation Test
    pool_test = np.vstack((X_test, Y_test))
    n_pool = n_test + m_test
    
    # Store permutation results: (n_permutations, K)
    perm_stats_list = []
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(n_pool)
        p_pool = pool_test[perm_idx]
        p_X = p_pool[:n_test]
        p_Y = p_pool[n_test:]
        
        p_stats = compute_spectral_statistic_grid(p_X, p_Y, Z, lambdas, h_base, h_mults)
        perm_stats_list.append(p_stats)
        
    perm_stats = np.array(perm_stats_list)
    
    # 5. Thresholds and Rejection
    # R: thres <- apply(perm_stats,2,sort)[ceiling((1-alpha/K)*num_perm),]
    # R: err_flg <- min(as.numeric(stat < thres)) 
    # If any stat >= thres, (stat < thres) is False, as.numeric is 0, min is 0. Return 0 (Reject).
    
    # Calculate adjusted alpha
    alpha_adj = alpha / K
    
    p_values = []
    for i in range(K):
        # Calculate threshold for this statistic
        # Quantile: 1 - alpha_adj
        # We sort perm_stats[:, i] and find the value at (1 - alpha_adj)
        sorted_perms = np.sort(perm_stats[:, i])
        idx_thres = int(np.ceil((1 - alpha_adj) * n_permutations))
        if idx_thres >= n_permutations: idx_thres = n_permutations - 1
        threshold = sorted_perms[idx_thres]
        
        if obs_stats[i] >= threshold:
            p_values.append(0.0) # Reject
        else:
            # Approximate p-value
            count = np.sum(perm_stats[:, i] >= obs_stats[i])
            p_values.append((count + 1) / (n_permutations + 1))
            
    # If we rejected for any i, we reject globally.
    min_p = np.min(p_values)
    # Bonferroni correction
    adj_p = min(K * min_p, 1.0)
    
    return adj_p