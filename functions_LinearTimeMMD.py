"""
Linear Time Kernel Two-Sample Test Functions
Based on Section 6 of "A Kernel Two-Sample Test" (Gretton et al., 2012)
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm

def get_median_bandwidth(X, Y, max_samples=1000):
    """
    Computes the median heuristic for the Gaussian kernel bandwidth.
    Subsamples if the dataset is too large to maintain O(m) overall complexity.
    """
    Z = np.vstack((X, Y))
    n_samples = Z.shape[0]
    # Subsample for bandwidth selection to keep it fast and memory-efficient
    if n_samples > max_samples:
        idx = np.random.choice(n_samples, max_samples, replace=False)
        Z_sub = Z[idx]
    else:
        Z_sub = Z
        
    # Median of squared Euclidean distances
    dists = pdist(Z_sub, 'sqeuclidean')  
    h = np.median(dists)
    if h == 0 or np.isnan(h):
        h = 1.0
    return h

def compute_linear_time_mmd(X, Y, bw=None):
    """
    Computes the linear time MMD^2 statistic and its asymptotic p-value.
    Assumes equal sample sizes. If not, truncates to min(m, n).
    
    Returns:
        mmd2_l: The linear time MMD^2 statistic.
        p_value: The asymptotic p-value based on the Gaussian approximation.
        h_i: The array of evaluated h(z_{2i-1}, z_{2i}) terms.
    """
    m = min(X.shape[0], Y.shape[0])
    m2 = m // 2
    
    if m2 < 2:
        raise ValueError("Sample size too small for linear time MMD.")
        
    # Truncate to ensure we have exactly m2 independent pairs
    X = X[:2*m2]
    Y = Y[:2*m2]
    
    if bw is None:
        bw = get_median_bandwidth(X, Y)
        
    # Extract pairs (x_{2i-1}, x_{2i}) and (y_{2i-1}, y_{2i})
    X1 = X[0::2]
    X2 = X[1::2]
    Y1 = Y[0::2]
    Y2 = Y[1::2]
    
    # Compute squared Euclidean distances for the pairs
    dist_xx = np.sum((X1 - X2)**2, axis=1)
    dist_yy = np.sum((Y1 - Y2)**2, axis=1)
    dist_xy1 = np.sum((X1 - Y2)**2, axis=1)
    dist_xy2 = np.sum((X2 - Y1)**2, axis=1)
    
    # Gaussian kernel: k(x, y) = exp(-||x-y||^2 / h)
    k_xx = np.exp(-dist_xx / bw)
    k_yy = np.exp(-dist_yy / bw)
    k_xy1 = np.exp(-dist_xy1 / bw)
    k_xy2 = np.exp(-dist_xy2 / bw)
    
    # h(z1, z2) = k(x1, x2) + k(y1, y2) - k(x1, y2) - k(x2, y1)
    h_i = k_xx + k_yy - k_xy1 - k_xy2
    
    mmd2_l = np.mean(h_i)
    
    # Estimate variance of h_i
    # Using sample variance provides better finite-sample calibration for the Z-test
    var_h = np.var(h_i, ddof=1)
    
    # Standard error of the mean
    se = np.sqrt(var_h / m2)
    
    if se < 1e-12:
        if mmd2_l > 1e-12:
            p_value = 0.0
        else:
            p_value = 1.0
    else:
        # Asymptotic normal test (one-sided, since MMD^2 >= 0 under H0)
        z_stat = mmd2_l / se
        p_value = 1.0 - norm.cdf(z_stat)
        
    return mmd2_l, p_value, h_i