import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# Unified kernel matrix computation (based on stacked samples)
# ------------------------------------------------------------
def _merged_kernel_matrix(X, Y, sigma, kernel_type):
    """
    Stack X and Y vertically and compute the full N x N kernel matrix.
    Supports three kernel types: 'gaussian', 'imq', 'laplacian'.
    """
    Z = np.vstack([X, Y])
    
    if kernel_type == 'gaussian':
        dists = pdist(Z, metric='euclidean')
        K = squareform(np.exp(-(dists ** 2) / sigma))
        np.fill_diagonal(K, 1.0)                 
    elif kernel_type == 'imq':
        dists_sq = pdist(Z, metric='sqeuclidean')
        K = squareform(1.0 / np.sqrt(dists_sq + sigma ** 2))
        np.fill_diagonal(K, 1.0 / sigma)         
    elif kernel_type == 'laplacian':
        dists = pdist(Z, metric='cityblock')
        K = squareform(np.exp(-dists / sigma))
        np.fill_diagonal(K, 1.0)                 
    else:
        raise ValueError("Unknown kernel_type")
    return K

def full_matrix_gaussian(X, Y, sigma):
    return _merged_kernel_matrix(X, Y, sigma, 'gaussian')

def full_matrix_imq(X, Y, sigma):
    return _merged_kernel_matrix(X, Y, sigma, 'imq')

def full_matrix_laplacian(X, Y, sigma):
    return _merged_kernel_matrix(X, Y, sigma, 'laplacian')


# ------------------------------------------------------------
# Efficient MMD statistic computation (directly from low-rank decomposition)
# ------------------------------------------------------------
def _mmd_statistic_from_lowrank(U_d, lambda_d, m, n):
    """
    Given low-rank decomposition K ≈ U_d @ diag(lambda_d) @ U_d.T,
    compute the unbiased MMD statistic directly without building the N x N matrix.
    """
    U_X = U_d[:m, :]          
    U_Y = U_d[m:, :]          
    lam = lambda_d            

    s_X = U_X.sum(axis=0)     
    s_Y = U_Y.sum(axis=0)

    sqnorm_X = (U_X ** 2).sum(axis=0)   
    sqnorm_Y = (U_Y ** 2).sum(axis=0)

    sum_K_XX = np.dot(lam, s_X ** 2)
    sum_K_YY = np.dot(lam, s_Y ** 2)
    sum_K_XY = np.dot(lam, s_X * s_Y)

    trace_K_XX = np.dot(lam, sqnorm_X)
    trace_K_YY = np.dot(lam, sqnorm_Y)

    if m > 1:
        term1 = (sum_K_XX - trace_K_XX) / (m * (m - 1))
    else:
        term1 = 0.0
    if n > 1:
        term2 = (sum_K_YY - trace_K_YY) / (n * (n - 1))
    else:
        term2 = 0.0
    term3 = 2.0 * sum_K_XY / (m * n)

    N = m + n
    stat = N * (term1 + term2 - term3)
    return stat


# ------------------------------------------------------------
# Multiple Kernel Test: Joint p-value computation
# ------------------------------------------------------------
def compute_joint_p_value(X, Y, d_list, sigma_list, p_hat, n_bootstrap, kernel_type_list):
    """
    Compute p-value for the aggregated test statistic across multiple kernels.
    
    Parameters:
    - d_list: list of truncation dimensions [d_1, d_2, ...] for each kernel
    - sigma_list: list of bandwidths [sigma_1, sigma_2, ...] for each kernel
    - kernel_type_list: list of kernel types ['gaussian', 'laplacian', ...] for each kernel
    """
    m, n = X.shape[0], Y.shape[0]
    N = m + n
    
    total_stat = 0.0
    S_list = []
    
    # Iterate over sigma, d, and kernel_type simultaneously
    for sigma, d, ktype in zip(sigma_list, d_list, kernel_type_list):
        K = _merged_kernel_matrix(X, Y, sigma, ktype)
        # Ensure k is less than N-1 for eigsh
        k_eig = min(d, N - 2)
        eigen_val, eigen_vec = eigsh(K, k=k_eig, which='LM')
        
        idx = np.argsort(eigen_val)[::-1]
        eigen_val = eigen_val[idx]
        eigen_vec = eigen_vec[:, idx]
        
        # Keep exactly d components
        eigen_val = eigen_val[:d]
        eigen_vec = eigen_vec[:, :d]
        
        # Accumulate the unbiased MMD statistic for kernel c
        stat_c = _mmd_statistic_from_lowrank(eigen_vec, eigen_val, m, n)
        total_stat += stat_c
        
        # Construct the evaluation matrix S_c = U_d * sqrt(Lambda_d)
        S_c = eigen_vec * np.sqrt(eigen_val)
        S_list.append(S_c)
        
    # Construct joint evaluation matrix V_gamma
    S_joint = np.hstack(S_list)
    centered_S_joint = S_joint - S_joint.mean(axis=0, keepdims=True)
    
    # Empirical joint cross-covariance matrix
    C_hat = (centered_S_joint.T @ centered_S_joint) / (N - 1)
    Gamma_hat = C_hat / (p_hat * (1 - p_hat))
    
    # Ensure positive definiteness
    try:
        np.linalg.cholesky(Gamma_hat)
    except np.linalg.LinAlgError:
        Gamma_hat += 1e-8 * np.eye(Gamma_hat.shape[0])
        
    d_total = S_joint.shape[1]
    mean_vec = np.zeros(d_total)
    
    # Parametric bootstrap for the joint null distribution
    boot_samples = multivariate_normal.rvs(mean=mean_vec, cov=Gamma_hat, size=n_bootstrap)
    if boot_samples.ndim == 1:
        stat_kerb = boot_samples ** 2 - np.trace(Gamma_hat)
    else:
        stat_kerb = np.sum(boot_samples ** 2, axis=1) - np.trace(Gamma_hat)
        
    pvalue = np.mean(total_stat < stat_kerb)
    return pvalue



# ------------------------------------------------------------
# Adaptive selection of d for Multiple Kernels
# ------------------------------------------------------------
def select_d_ratio_joint(X, Y, d_bar, sigma_list, p_hat, kernel_type_list):
    """
    Select optimal truncation dimension d_c independently for each kernel 
    to maximize the individual Signal-to-Noise Ratio (SNR).
    """
    m, n = X.shape[0], Y.shape[0]
    N = m + n
    
    best_d_list = []
    # Iterate over sigma and kernel_type simultaneously
    for sigma, ktype in zip(sigma_list, kernel_type_list):
        K = _merged_kernel_matrix(X, Y, sigma, ktype)
        k_eig = min(d_bar, N - 2)
        eigen_val, eigen_vec = eigsh(K, k=k_eig, which='LM')
        idx = np.argsort(eigen_val)[::-1]
        eigen_val = eigen_val[idx]
        eigen_vec = eigen_vec[:, idx]
        
        S = eigen_vec * np.sqrt(eigen_val)
        centered_S = S - S.mean(axis=0, keepdims=True)
        C_hat_full = (centered_S.T @ centered_S) / (N - 1)
        Gamma_hat_full = C_hat_full / (p_hat * (1 - p_hat))
        Gamma_hat_full += 1e-10 * np.eye(d_bar)
        
        snr_values = []
        for d in range(1, d_bar + 1):
            U_d = eigen_vec[:, :d]
            lambda_d = eigen_val[:d]
            signal = _mmd_statistic_from_lowrank(U_d, lambda_d, m, n)
            Gamma_d = Gamma_hat_full[:d, :d]
            variance_proxy = np.trace(Gamma_d @ Gamma_d)
            noise = np.sqrt(variance_proxy) if variance_proxy > 0 else 1e-12
            snr = signal / noise
            snr_values.append(snr)
            
        best_d_list.append(np.argmax(snr_values) + 1)
        
    return best_d_list