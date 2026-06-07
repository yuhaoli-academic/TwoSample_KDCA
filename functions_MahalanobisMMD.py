import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import chi2

def gaussian_kernel(X, Y, sigma):
    """Gaussian kernel matrix."""
    D_sq = cdist(X, Y, 'sqeuclidean')
    return np.exp(-D_sq / (2 * sigma**2))

def laplace_kernel(X, Y, sigma):
    """Laplace kernel matrix."""
    D = cdist(X, Y, 'cityblock')
    return np.exp(-D / sigma)

def center_kernel_matrix(K):
    """Centers the kernel matrix K."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def compute_mmd_unbiased(X, Y, sigma, kernel_type):
    """Computes the unbiased MMD^2 statistic."""
    m = X.shape[0]
    n = Y.shape[0]
    
    if kernel_type == 'gaussian':
        K_XX = gaussian_kernel(X, X, sigma)
        K_YY = gaussian_kernel(Y, Y, sigma)
        K_XY = gaussian_kernel(X, Y, sigma)
    elif kernel_type == 'laplace':
        K_XX = laplace_kernel(X, X, sigma)
        K_YY = laplace_kernel(Y, Y, sigma)
        K_XY = laplace_kernel(X, Y, sigma)
    else:
        raise ValueError("Unknown kernel type")

    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    term1 = np.sum(K_XX) / (m * (m - 1))
    term2 = np.sum(K_YY) / (n * (n - 1))
    term3 = 2 * np.sum(K_XY) / (m * n)
    
    return term1 + term2 - term3

def MMMD_test(X, Y,  n_bootstrap=500, kernel_configs=None):
    """
    Performs the Mahalanobis Aggregated MMD (MMMD) test.
    Corrected implementation based on Equations 4.1-4.3.
    """
    m, d = X.shape
    n = Y.shape[0]
    N = m + n
    rho = m / N
    
    # 1. Bandwidth Selection
    Z_pool = np.vstack((X, Y))
    dists = pdist(Z_pool, 'euclidean')
    med_dist = np.median(dists)
    
    if kernel_configs is None:
        multipliers = [0.5, 1/np.sqrt(2), 1.0, np.sqrt(2), 2.0]
        kernel_configs = [{'type': 'gaussian', 'sigma': med_dist * mul} for mul in multipliers]
    
    r = len(kernel_configs)
    
    # 2. Compute Centered Kernel Matrices for X
    # We store K_c (H K H) and K_hat_c (H K H / m)
    K_c_list = []
    K_hat_c_list = []
    
    for config in kernel_configs:
        k_type = config['type']
        sigma = config['sigma']
        
        if k_type == 'gaussian':
            K_X = gaussian_kernel(X, X, sigma)
        elif k_type == 'laplace':
            K_X = laplace_kernel(X, X, sigma)
            
        K_c = center_kernel_matrix(K_X)
        K_c_list.append(K_c)
        K_hat_c_list.append(K_c / m) # Eq 4.1: K_hat_c = C K C / m
        
    # 3. Estimate Covariance Matrix Sigma_hat (Eq 2.14)
    scale_factor = 2 / (rho**2 * (1 - rho)**2 * m**2)
    Sigma_hat = np.zeros((r, r))
    
    for i in range(r):
        for j in range(r):
            cov_val = scale_factor * np.sum(K_c_list[i] * K_c_list[j])
            Sigma_hat[i, j] = cov_val
            
    # Regularization
    reg_val = 1e-5 * np.min(np.diag(Sigma_hat))
    if reg_val < 0: reg_val = 0
    Sigma_reg = Sigma_hat + reg_val * np.eye(r)
    
    try:
        Sigma_inv = np.linalg.inv(Sigma_reg)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_reg)

    # 4. Compute Observed Statistic (Eq 3.6 scaling)
    # Statistic is N * MMD^2 vector
    MMD_vec = np.zeros(r)
    for i, config in enumerate(kernel_configs):
        mmd_val = compute_mmd_unbiased(X, Y, config['sigma'], config['type'])
        MMD_vec[i] = N * mmd_val
        
    T_obs = MMD_vec @ Sigma_inv @ MMD_vec
    
    # 5. Multiplier Bootstrap (Eq 4.2)
    # Z ~ N(0, 1/(rho(1-rho)) I)
    var_z = 1 / (rho * (1 - rho))
    Z_mat = np.random.randn(m, n_bootstrap) * np.sqrt(var_z)
    
    V_boot = np.zeros((r, n_bootstrap))
    
    for i in range(r):
        K_hat_c = K_hat_c_list[i]
        
        # Quadratic form: diag(Z^T K_hat Z) = sum(Z * (K @ Z))
        quad_form = np.sum(Z_mat * (K_hat_c @ Z_mat), axis=0)
        
        # Trace term: (1/(rho(1-rho))) * Tr(K_hat)
        trace_term = var_z * np.trace(K_hat_c)
        
        V_boot[i, :] = quad_form - trace_term
        
    # 6. Compute Bootstrap Mahalanobis Distances (Eq 4.3)
    S_boot = np.sum((V_boot.T @ Sigma_inv) * V_boot.T, axis=1)
    
    # 7. P-value
    p_val = np.mean(S_boot >= T_obs)

    
    return  p_val