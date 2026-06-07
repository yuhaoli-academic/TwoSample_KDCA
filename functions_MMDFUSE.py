import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

def kernel_matrix(pairwise_matrix, l, kernel, bandwidth):
    """
    Compute kernel matrix for a given kernel and bandwidth.
    
    Parameters:
    -----------
    pairwise_matrix : np.ndarray
        Matrix of pairwise distances.
    l : str
        Distance metric type: 'l1' or 'l2'.
    kernel : str
        Kernel type: 'gaussian', 'laplace', 'imq', etc.
    bandwidth : float
        Kernel bandwidth.
        
    Returns:
    --------
    K : np.ndarray
        The kernel matrix.
    """
    d = pairwise_matrix / bandwidth
    
    if kernel == "gaussian" and l == "l2":
        K = np.exp(-(d**2) / 2)
    elif kernel == "laplace" and l == "l1":
        K = np.exp(-d * np.sqrt(2))
    elif kernel == "rq" and l == "l2":
        # Rational Quadratic kernel
        alpha = 0.5
        K = (1 + d**2 / (2 * alpha)) ** (-alpha)
    elif kernel == "imq" and l == "l2":
        # Inverse Multi-Quadratic
        K = (1 + d**2) ** (-0.5)
    elif (kernel == "matern_0.5_l1" and l == "l1") or (kernel == "matern_0.5_l2" and l == "l2"):
        K = np.exp(-d)
    elif (kernel == "matern_1.5_l1" and l == "l1") or (kernel == "matern_1.5_l2" and l == "l2"):
        K = (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
    elif (kernel == "matern_2.5_l1" and l == "l1") or (kernel == "matern_2.5_l2" and l == "l2"):
        K = (1 + np.sqrt(5) * d + 5/3 * d**2) * np.exp(-np.sqrt(5) * d)
    elif (kernel == "matern_3.5_l1" and l == "l1") or (kernel == "matern_3.5_l2" and l == "l2"):
        K = (1 + np.sqrt(7) * d + 14/5 * d**2 + 7*np.sqrt(7)/15 * d**3) * np.exp(-np.sqrt(7) * d)
    elif (kernel == "matern_4.5_l1" and l == "l1") or (kernel == "matern_4.5_l2" and l == "l2"):
        K = (1 + 3*d + 27/7 * d**2 + 18/7 * d**3 + 27/35 * d**4) * np.exp(-3*d)
    else:
        raise ValueError(f'Invalid combination of kernel "{kernel}" and distance "{l}".')
        
    return K

def mmdfuse(X, Y,  kernels=("laplace", "gaussian"), lambda_multiplier=1, 
            number_bandwidths=10, number_permutations=500, seed=None):
    """
    Two-Sample MMD-FUSE test (NumPy implementation).
    
    Parameters:
    -----------
    X : np.ndarray
        Sample from distribution P (shape m, d).
    Y : np.ndarray
        Sample from distribution Q (shape n, d).
    kernels : tuple
        List of kernel strings.
    lambda_multiplier : float
        Multiplier for the regularization parameter.
    number_bandwidths : int
        Number of bandwidths to try per kernel.
    number_permutations : int
        Number of permutations for the null distribution.
    return_p_val : bool
        If True, returns p-value as well.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    reject : int
        1 if rejected, 0 otherwise.
    p_val : float, optional
        The p-value of the test.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Ensure Y is the smaller sample (convention from original code)
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
    
    m, d = X.shape
    n, _ = Y.shape
    N = m + n
    
    # Assertions (checking inputs)
    assert n >= 2 and m >= 2

    
    # Separate kernels by distance metric requirement
    all_kernels_l1 = ("laplace", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", 
                      "matern_3.5_l1", "matern_4.5_l1")
    all_kernels_l2 = ("imq", "rq", "gaussian", "matern_0.5_l2", "matern_1.5_l2", 
                      "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]
    
    # Setup Permutations
    B = number_permutations
    
    # Generate random permutation indices (B+1, N)
    # Note: JAX code generates permutations "independently" per row. 
    # Numpy argsort on random data achieves this.
    rand_vals = np.random.rand(B + 1, N)
    idx = np.argsort(rand_vals, axis=1)
    
    # Construct witness vectors for permutations
    # V11: 1s for X, -1s for Y
    v11 = np.concatenate((np.ones(m), -np.ones(n)))
    V11 = v11[idx] # (B+1, N)
    
    # V10: 1s for X, 0s for Y
    v10 = np.concatenate((np.ones(m), np.zeros(n)))
    V10 = v10[idx]
    
    # V01: 0s for X, -1s for Y
    v01 = np.concatenate((np.zeros(m), -np.ones(n)))
    V01 = v01[idx]
    
    # The last permutation (index B) should be the original order (identity)

    idx[B] = np.arange(N)
    V11[B] = v11
    V10[B] = v10
    V01[B] = v01
    
    # Transpose for matrix multiplication: (N, B+1)
    V11 = V11.T
    V10 = V10.T
    V01 = V01.T
    
    # MMD Estimates Storage
    M_list = []
    
    # Combine data
    Z = np.vstack((X, Y))
    
    # Helper to compute bandwidths
    def get_bandwidths(distances, k):
        # Remove zeros for quantile calculation logic robustness
        # JAX code logic: distances + (distances==0)*median
        # We just sort and pick quantiles.
        dd = np.sort(distances)
        # 5th and 95th percentiles
        lambda_min = dd[int(len(dd) * 0.05)] / 2
        lambda_max = dd[int(len(dd) * 0.95)] * 2
        return np.linspace(lambda_min, lambda_max, k)
    
    # Process L1 kernels
    if len(kernels_l1) > 0:
        # Compute L1 distance matrix
        # cdist is efficient (m+n, m+n)
        dists = cdist(Z, Z, 'cityblock')
        # Extract upper triangle for bandwidth selection
        triu_dists = dists[np.triu_indices(N, k=1)]
        bandwidths = get_bandwidths(triu_dists, number_bandwidths)
        
        for kernel in kernels_l1:
            for bw in bandwidths:
                K = kernel_matrix(dists, "l1", kernel, bw)
                np.fill_diagonal(K, 0) # Zero diagonal for U-statistic
                
                # Normalizer
                unscaled_std = np.sqrt(np.sum(K**2))
                
                # Compute MMD stats for all permutations
                # Formula derived from JAX code (MMDAgg Appendix C style)
                # V.T @ K @ V logic vectorized: sum(V * (K @ V), axis=0)
                
                KV10 = K @ V10
                KV01 = K @ V01
                KV11 = K @ V11
                
                term1 = np.sum(V10 * KV10, axis=0) * (n - m + 1) * (n - 1) / (m * (m - 1))
                term2 = np.sum(V01 * KV01, axis=0) * (m - n + 1) / m
                term3 = np.sum(V11 * KV11, axis=0) * (n - 1) / m
                
                # Normalize
                stat = (term1 + term2 + term3) / (unscaled_std * np.sqrt(n * (n - 1)))
                M_list.append(stat)
                
    # Process L2 kernels
    if len(kernels_l2) > 0:
        # Compute L2 distance matrix
        dists = cdist(Z, Z, 'euclidean')
        triu_dists = dists[np.triu_indices(N, k=1)]
        bandwidths = get_bandwidths(triu_dists, number_bandwidths)
        
        for kernel in kernels_l2:
            for bw in bandwidths:
                K = kernel_matrix(dists, "l2", kernel, bw)
                np.fill_diagonal(K, 0)
                
                unscaled_std = np.sqrt(np.sum(K**2))
                
                KV10 = K @ V10
                KV01 = K @ V01
                KV11 = K @ V11
                
                term1 = np.sum(V10 * KV10, axis=0) * (n - m + 1) * (n - 1) / (m * (m - 1))
                term2 = np.sum(V01 * KV01, axis=0) * (m - n + 1) / m
                term3 = np.sum(V11 * KV11, axis=0) * (n - 1) / m
                
                stat = (term1 + term2 + term3) / (unscaled_std * np.sqrt(n * (n - 1)))
                M_list.append(stat)
                
    # Stack M: Shape (num_total_kernels, B+1)
    M = np.array(M_list)
    N_kernels = M.shape[0]
    
    # Aggregation (FUSE)
    # lambda parameter
    lam = lambda_multiplier * np.sqrt(n * (n - 1))
    
    # Log-sum-exp aggregation
    # b=1/N normalizes the uniform distribution over kernels implicitly
    # axis=0 aggregates over kernels
    all_statistics = logsumexp(lam * M, axis=0, b=1/N_kernels)
    
    original_statistic = all_statistics[-1]
    
    # P-value calculation
    # Proportion of permuted stats >= original stat
    # Use >= to be conservative, matching JAX logic
    p_val = np.mean(all_statistics >= original_statistic)
    
    
    return p_val