import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm

def gaussian_kernel_matrix(X, Y, sigma):
    """
    Computes the Gaussian kernel matrix between X and Y.
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    # Compute squared Euclidean distances
    pairwise_dists_sq = cdist(X, Y, 'sqeuclidean')
    return np.exp(-pairwise_dists_sq / (2 * sigma**2))

def xMMD_test(X, Y, alpha=0.05):
    """
    Performs the cross-MMD two-sample test (xMMD) as proposed by Shekhar et al.
    
    The test uses sample splitting and studentization to achieve a standard 
    Gaussian limiting distribution under the null, avoiding the need for 
    computationally expensive permutations.
    
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
    n = X.shape[0]
    m = Y.shape[0]
    
    # 1. Sample Splitting
    # Balanced split: n1 = floor(n/2), m1 = floor(m/2)
    n1 = n // 2
    m1 = m // 2
    
    X1, X2 = X[:n1], X[n1:]
    Y1, Y2 = Y[:m1], Y[m1:]
    
    n2 = X2.shape[0]
    m2 = Y2.shape[0]
    
    if n2 == 0 or m2 == 0 or n1 == 0 or m1 == 0:
        raise ValueError("Sample size too small for splitting.")

    # 2. Bandwidth Selection (Median Heuristic)
    # The paper supports using the full sample for bandwidth selection (Theorem 5).
    Z = np.vstack((X, Y))
    dists = pdist(Z, 'euclidean')
    sigma = np.median(dists)
    if sigma == 0:
        sigma = 1.0

    # 3. Compute Cross-Kernel Blocks
    # We only need the cross-blocks (X1 vs X2, X1 vs Y2, etc.), not the full matrix.
    # This reduces computation roughly by half compared to full MMD.
    K_X1_X2 = gaussian_kernel_matrix(X1, X2, sigma)
    K_X1_Y2 = gaussian_kernel_matrix(X1, Y2, sigma)
    K_Y1_X2 = gaussian_kernel_matrix(Y1, X2, sigma)
    K_Y1_Y2 = gaussian_kernel_matrix(Y1, Y2, sigma)
    
    # 4. Compute U-statistics components
    # U_X,i = mean(k(X_i, X2)) - mean(k(X_i, Y2))
    U_X = np.mean(K_X1_X2, axis=1) - np.mean(K_X1_Y2, axis=1)
    
    # U_Y,j = mean(k(Y_j, X2)) - mean(k(Y_j, Y2))
    U_Y = np.mean(K_Y1_X2, axis=1) - np.mean(K_Y1_Y2, axis=1)
    
    # 5. Compute Statistic
    # xMMD^2 = mean(U_X) - mean(U_Y)
    xMMD_sq = np.mean(U_X) - np.mean(U_Y)
    
    # 6. Studentization
    # Variance terms (using biased variance estimator with n1 denominator as per Eq 4 in paper)
    # sigma^2_X = 1/n1 sum(U_X - mean(U_X))^2
    sigma_sq_X = np.mean((U_X - np.mean(U_X))**2)
    sigma_sq_Y = np.mean((U_Y - np.mean(U_Y))**2)
    
    # Total variance of the statistic
    # sigma^2 = 1/n1 * sigma^2_X + 1/m1 * sigma^2_Y
    stat_var = sigma_sq_X / n1 + sigma_sq_Y / m1
    
    # 7. Compute Standardized Statistic
    if stat_var < 1e-10:
        T = 0.0
    else:
        T = xMMD_sq / np.sqrt(stat_var)
        
    # 8. P-value and Decision
    # Under H0, T ~ N(0, 1). Right-tailed test.
    p_value = 1 - norm.cdf(T)
    reject = int(p_value <= alpha)
    
    return reject, p_value