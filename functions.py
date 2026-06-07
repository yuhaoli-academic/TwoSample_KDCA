import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# Unified kernel matrix computation (based on stacked samples)
# ------------------------------------------------------------
def _merged_kernel_matrix(X, Y, sigma, kernel_type='gaussian'):
    """
    Stack X and Y vertically and compute the full N×N kernel matrix.
    Supports three kernel types: 'gaussian', 'imq', 'laplacian'.

    Note: pdist returns distances between distinct pairs. squareform
    reconstructs a symmetric matrix with zeros on the diagonal.
    Because we apply the kernel *before* squareform, the diagonal is left as 0.
    We must explicitly set it to the correct value:
        - 1.0 for Gaussian and Laplacian
        - 1.0/sigma for IMQ
    """
    Z = np.vstack([X, Y])
    
    if kernel_type == 'gaussian':
        dists = pdist(Z, metric='euclidean')
        K = squareform(np.exp(-(dists ** 2) / sigma))
        np.fill_diagonal(K, 1.0)                 # Correct diagonal for Gaussian
    elif kernel_type == 'imq':
        dists_sq = pdist(Z, metric='sqeuclidean')
        K = squareform(1.0 / np.sqrt(dists_sq + sigma ** 2))
        np.fill_diagonal(K, 1.0 / sigma)         # Correct diagonal for IMQ
    elif kernel_type == 'laplacian':
        dists = pdist(Z, metric='cityblock')
        K = squareform(np.exp(-dists / sigma))
        np.fill_diagonal(K, 1.0)                 # Correct diagonal for Laplacian
    else:
        raise ValueError("Unknown kernel_type")
    return K


# Keep original function names for backward compatibility
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
    compute the unbiased MMD statistic directly without building the N×N matrix.
    """
    # Separate features for the two samples
    U_X = U_d[:m, :]          # m × d
    U_Y = U_d[m:, :]          # n × d
    lam = lambda_d            # (d,)

    # Column sums: s_X_k = sum_i U_X[i,k], s_Y_k = sum_j U_Y[j,k]
    s_X = U_X.sum(axis=0)     # (d,)
    s_Y = U_Y.sum(axis=0)

    # Sum of squared columns (used for trace computation)
    sqnorm_X = (U_X ** 2).sum(axis=0)   # (d,)
    sqnorm_Y = (U_Y ** 2).sum(axis=0)

    # Sum of all elements in K_XX: Σ_k λ_k * (s_X_k)^2
    sum_K_XX = np.dot(lam, s_X ** 2)
    # Sum of all elements in K_YY
    sum_K_YY = np.dot(lam, s_Y ** 2)
    # Sum of all elements in K_XY: Σ_k λ_k * s_X_k * s_Y_k
    sum_K_XY = np.dot(lam, s_X * s_Y)

    # Trace of K_XX: Σ_k λ_k * (Σ_i U_X[i,k]^2)
    trace_K_XX = np.dot(lam, sqnorm_X)
    trace_K_YY = np.dot(lam, sqnorm_Y)

    # Unbiased MMD estimator (scaled by N)
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
# Compute p-value for a given d (using low-rank decomposition)
# ------------------------------------------------------------
def compute_p_value_for_d(U_d, lambda_d, m, n, p_hat, n_bootstrap):
    """
    Compute p-value directly using the first d eigenvectors and eigenvalues.
    The original X, Y, sigma parameters are already encoded in U_d and lambda_d.
    """
    N = m + n
    d = len(lambda_d)
    
    # 1. Unbiased MMD statistic
    stat_ker = _mmd_statistic_from_lowrank(U_d, lambda_d, m, n)
    
    # 2. Build score matrix S (N×d) and compute asymptotic covariance Gamma_hat
    S = U_d * np.sqrt(lambda_d)          # (N, d)
    centered_S = S - S.mean(axis=0, keepdims=True)
    C_hat = (centered_S.T @ centered_S) / (N - 1)
    Gamma_hat = C_hat / (p_hat * (1 - p_hat))
    
    # Ensure positive definiteness
    try:
        np.linalg.cholesky(Gamma_hat)
    except np.linalg.LinAlgError:
        Gamma_hat += 1e-8 * np.eye(d)
    
    # 3. Parametric bootstrap
    mean_vec = np.zeros(d)
    boot_samples = multivariate_normal.rvs(mean=mean_vec, cov=Gamma_hat, size=n_bootstrap)
    if boot_samples.ndim == 1:
        stat_kerb = boot_samples ** 2 - np.trace(Gamma_hat)
    else:
        stat_kerb = np.sum(boot_samples ** 2, axis=1) - np.trace(Gamma_hat)
    
    pvalue_ker = np.mean(stat_ker < stat_kerb)
    return pvalue_ker


# ------------------------------------------------------------
# Adaptive selection of d (fully vectorized, avoids redundant computation)
# ------------------------------------------------------------
def select_d(X, Y, d_bar, sigma, p_hat, n_bootstrap):
    m, n = X.shape[0], Y.shape[0]
    N = m + n
    
    # 1. Compute full kernel matrix once and obtain top d_bar eigenpairs
    K = _merged_kernel_matrix(X, Y, sigma, 'gaussian')
    eigen_val, eigen_vec = eigsh(K, k=d_bar, which='LM')
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigen_val)[::-1]
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:, idx]
    
    # 2. Compute p-value for each candidate d, reusing the decomposition
    p_values = []
    for d in range(1, d_bar + 1):
        U_d = eigen_vec[:, :d]
        lambda_d = eigen_val[:d]
        p_val = compute_p_value_for_d(U_d, lambda_d, m, n, p_hat, n_bootstrap)
        p_values.append(p_val)
    
    best_j = np.argmin(p_values) + 1
    return best_j

def select_d_ratio(X, Y, d_bar, sigma, p_hat):
    m, n = X.shape[0], Y.shape[0]
    N = m + n
    
    # 1. Compute full kernel matrix once
    K = _merged_kernel_matrix(X, Y, sigma, 'gaussian')
    
    # 2. Compute top d_bar eigenpairs
    eigen_val, eigen_vec = eigsh(K, k=d_bar, which='LM')
    idx = np.argsort(eigen_val)[::-1]
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:, idx]
    
    # 3. Precompute centered score matrix and asymptotic covariance for max d_bar
    S = eigen_vec[:, :d_bar] * np.sqrt(eigen_val[:d_bar])
    centered_S = S - S.mean(axis=0, keepdims=True)
    C_hat_full = (centered_S.T @ centered_S) / (N - 1)
    Gamma_hat_full = C_hat_full / (p_hat * (1 - p_hat))
    
    # Numerical safeguard for near-singular trailing directions
    Gamma_hat_full += 1e-10 * np.eye(d_bar)
    
    snr_values = []
    for d in range(1, d_bar + 1):
        U_d = eigen_vec[:, :d]
        lambda_d = eigen_val[:d]
        
        # Compute signal: unbiased MMD statistic for dimension d
        signal = _mmd_statistic_from_lowrank(U_d, lambda_d, m, n)
        
        # Extract restricted covariance and compute EXACT null variance proxy
        Gamma_d = Gamma_hat_full[:d, :d]
        variance_proxy = np.trace(Gamma_d @ Gamma_d)  # tr(Γ_d^2)
        
        # Deflection coefficient denominator (standard deviation scale)
        noise = np.sqrt(variance_proxy) if variance_proxy > 0 else 1e-12
        
        # Exact SNR proxy for argmax
        snr = signal / noise
        snr_values.append(snr)
        
    best_j = np.argmax(snr_values) + 1
    return best_j

# ------------------------------------------------------------
# Final p-value computation for a specified j
# ------------------------------------------------------------
def p_value_calculation(X, Y, j, sigma, p_hat, n_bootstrap):
    m, n = X.shape[0], Y.shape[0]
    
    K = _merged_kernel_matrix(X, Y, sigma, 'gaussian')
    eigen_val, eigen_vec = eigsh(K, k=j, which='LM')
    
    idx = np.argsort(eigen_val)[::-1]
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:, idx]
    
    return compute_p_value_for_d(eigen_vec, eigen_val, m, n, p_hat, n_bootstrap)
