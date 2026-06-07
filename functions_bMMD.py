import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

def mmd_u_statistic(X, Y, sigma):
    """
    Computes the unbiased MMD^2 U-statistic for two samples X and Y.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        First sample.
    Y : ndarray, shape (m, d)
        Second sample.
    sigma : float
        Gaussian kernel bandwidth (kernel = exp(-||x-y||^2 / sigma)).

    Returns
    -------
    mmd_u : float
        Unbiased squared MMD estimate.
    """
    n = X.shape[0]
    m = Y.shape[0]

    # Compute kernel matrices
    K_XX = np.exp(-cdist(X, X, metric='sqeuclidean') / sigma)
    K_YY = np.exp(-cdist(Y, Y, metric='sqeuclidean') / sigma)
    K_XY = np.exp(-cdist(X, Y, metric='sqeuclidean') / sigma)

    # Zero out diagonals for unbiased estimate
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    sum_K_XX = np.sum(K_XX)
    sum_K_YY = np.sum(K_YY)
    sum_K_XY = np.sum(K_XY)

    mmd_u = (sum_K_XX / (n * (n - 1))) + \
            (sum_K_YY / (m * (m - 1))) - \
            (2 * sum_K_XY / (n * m))

    return mmd_u


def b_test_p_value(X, Y, sigma, block_size=None):
    """
    Performs the B-test for two-sample testing exactly as described in 
    Zaremba et al., "B-tests: Low Variance Kernel Two-Sample Tests".

    Parameters
    ----------
    X : ndarray, shape (n, d)
        First sample.
    Y : ndarray, shape (m, d)
        Second sample.
    sigma : float
        Gaussian kernel bandwidth.
    block_size : int or None, optional
        Size of each block. If None, uses floor(sqrt(min(n, m))) 
        corresponding to gamma = 0.5 in the paper.

    Returns
    -------
    p_value : float
        p-value of the test (one-sided, right tail).
    """
    # Use equal sample sizes by truncating to the smaller one
    n = X.shape[0]
    m = Y.shape[0]
    N = min(n, m)
    X = X[:N]
    Y = Y[:N]

    # Default block size heuristic from the paper: B = floor(n^gamma) with gamma=0.5
    if block_size is None:
        block_size = int(np.floor(np.sqrt(N)))

    # Ensure block size is at least 2 (otherwise MMD_u statistic is invalid)
    block_size = max(block_size, 2)

    # Number of blocks
    num_blocks = N // block_size

    # We need at least 2 blocks to estimate variance reliably
    if num_blocks < 2:
        raise ValueError(
            f"Not enough samples for B-test with block_size={block_size}. "
            f"Need at least {2 * block_size} samples, but got {N}. "
            "Reduce block_size or provide more data."
        )

    block_stats = np.zeros(num_blocks)

    # 1. Compute the block-wise MMD U-statistics
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        
        X_block = X[start:end]
        Y_block = Y[start:end]
        
        block_stats[i] = mmd_u_statistic(X_block, Y_block, sigma)

    # 2. B-test statistic: average of block statistics (Eq. 4 in paper)
    test_stat = np.mean(block_stats)

    # 3. Estimate variance under H0 using the sample variance of the block statistics
    # Var(mean) = Sample_Var(blocks) / num_blocks
    var_mean = np.var(block_stats, ddof=1) / num_blocks

    # 4. Compute p-value using asymptotic Normal distribution (Theorem 2.2)
    if var_mean <= 1e-15:
        # If variance is essentially zero, decide strictly based on test_stat
        p_value = 1.0 if test_stat <= 0 else 0.0
    else:
        z_score = test_stat / np.sqrt(var_mean)
        p_value = 1 - norm.cdf(z_score)

    return p_value