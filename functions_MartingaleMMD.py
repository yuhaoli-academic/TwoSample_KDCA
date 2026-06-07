import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm

def mMMD_test(X, Y, alpha=0.05):
    """
    Performs the Martingale MMD (mMMD) two-sample test.
    
    Implementation based on the logic from tests.py (mMMD_test_statistic class).
    
    Parameters:
    -----------
    X : np.ndarray
        Sample from distribution P (shape n, d).
    Y : np.ndarray
        Sample from distribution Q (shape m, d).
    alpha : float
        Significance level (default 0.05).
        
    Returns:
    --------
    reject : int
        1 if null hypothesis is rejected, 0 otherwise.
    p_value : float
        The p-value of the test.
    """
    # Handle sample sizes
    n = X.shape[0]
    m = Y.shape[0]
    
    # mMMD assumes paired samples (X_i, Y_i). 
    # If n != m, we take the minimum size to form pairs.
    N = min(n, m)
    if N < 2:
        return 0, 1.0

    X = X[:N]
    Y = Y[:N]
    
    # 1. Bandwidth Selection (Median Heuristic)
    Z_pool = np.vstack((X, Y))
    dists = pdist(Z_pool, 'euclidean')
    sigma = np.median(dists)
    if sigma == 0:
        sigma = 1.0
        
    # 2. Calculate kernel matrices
    # Using Gaussian kernel: exp(- dist^2 / (2 * sigma^2))
    K_XX = np.exp(-cdist(X, X, 'sqeuclidean') / (2 * sigma**2))
    K_YY = np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * sigma**2))
    K_XY = np.exp(-cdist(X, Y, 'sqeuclidean') / (2 * sigma**2))
    K_YX = K_XY.T
    
    # 3. Calculate the H matrix
    # H = K(x,x) - K(x,y) - K(y,x) + K(y,y)
    H_matrix = K_XX - K_XY - K_YX + K_YY
    
    # 4. Extract the lower triangle (excluding the diagonal)
    # np.tril(M, k=-1) keeps elements strictly below the main diagonal
    lower_triangle_H = np.tril(H_matrix, k=-1)
    
    # 5. Sum the lower triangle along the columns (axis 1)
    # This computes Sum_{j=0 to i-1} H_ij for each row i
    sum_H_j = np.sum(lower_triangle_H, axis=1)
    
    # 6. Calculate the weights
    # Row i (0-based) corresponds to i+1 in 1-based indexing.
    # The weight is 1/(i+1) for i >= 1.
    # sum_H_j[0] is always 0 (no elements below diagonal in first row).
    # We operate on sum_H_j[1:] which corresponds to rows 1 to N-1.
    # Weights should correspond to 1/(row_index + 1).
    # For row 1, weight is 1/2. For row N-1, weight is 1/N.
    # np.arange(2, N + 1) generates [2, ..., N], and we take the inverse.
    weights = (np.arange(2, N + 1).astype(float)**(-1))
    
    # 7. Calculate the weighted terms
    # Multiply the row sums (skipping the first row) by the weights
    weighted_sums = sum_H_j[1:] * weights
    
    # 8. Compute the self-normalized statistic
    numerator = np.sum(weighted_sums)
    denominator = np.sqrt(np.sum(weighted_sums**2))
    
    if denominator < 1e-10:
        statistic = 0.0
    else:
        statistic = numerator / denominator
        
    # 9. P-value and Decision
    # Under H0, statistic ~ N(0, 1). Right-tailed test.
    p_value = 1 - norm.cdf(statistic)
    reject = int(p_value <= alpha)
    
    return reject, p_value