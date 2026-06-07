import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import eigh

def full_kernel_matrix(Z, sigma):
    """
    Computes the full Gaussian kernel matrix for the pooled sample Z.
    """
    # Compute pairwise squared Euclidean distances
    pairwise_dists = pdist(Z, 'sqeuclidean')
    # Convert to square form
    D_sq = squareform(pairwise_dists)
    # Gaussian kernel
    K = np.exp(-D_sq / sigma)
    return K

def mmd2_u_statistic(K, m, n):
    """
    Computes the unbiased MMD^2_u statistic given the kernel matrix K 
    on the pooled sample [X, Y], where |X|=m, |Y|=n.
    """
    if m != n:
        raise ValueError("Gretton et al. 2009 Spectrum MMD strictly requires equal sample sizes (m == n) for the one-sample U-statistic.")
      
    # Extract blocks
    K_XX = K[:m, :m]
    K_YY = K[m:, m:]
    K_XY = K[:m, m:]
    
    # Sum of off-diagonal elements for XX and YY
    # np.sum(K_XX) includes diagonal, np.trace(K_XX) removes it
    sum_K_XX = np.sum(K_XX) - np.trace(K_XX)
    sum_K_YY = np.sum(K_YY) - np.trace(K_YY)
    
    # Sum of all elements for XY
    sum_K_XY = np.sum(K_XY)
    
    # Unbiased MMD^2 formula
    mmd2 = (sum_K_XX / (m * (m - 1))) + \
           (sum_K_YY / (n * (n - 1))) - \
           (2 * sum_K_XY / (m * n))
           
    return mmd2

def p_value_spectrum_mmd(X, Y, n_bootstrap=1000):
    """
    Performs the Spectrum MMD test (Spec) as described in Gretton et al.
    
    Parameters:
    -----------
    X : np.ndarray (m, d)
        Sample from distribution P.
    Y : np.ndarray (n, d)
        Sample from distribution Q.
    n_bootstrap : int
        Number of samples to generate for the null distribution approximation.
        
    Returns:
    --------
    p_value : float
        The p-value of the test.
    """
    m = X.shape[0]
    n = Y.shape[0]
    N = m + n
    
    # 1. Pool Data
    Z = np.vstack((X, Y))
    
    # 2. Determine Kernel Bandwidth (Median Heuristic)
    # Paper Section 4: "bandwidth set to the median distance between points in the aggregation"
    dists = pdist(Z, 'euclidean')
    sigma = np.median(dists)**2
    # Fallback for trivial data
    if sigma == 0:
        sigma = 1.0
    
    # 3. Compute Gram Matrix K
    K = full_kernel_matrix(Z, sigma)
    
    # 4. Compute Unbiased MMD Statistic
    # The paper discusses m * MMD^2_u convergence. We compute the raw statistic first.
    mmd2_u = mmd2_u_statistic(K, m, n)
    
    # 5. Compute Eigenvalues of Centered Gram Matrix
    # Centering matrix H = I - (1/N) 11^T
    # K_centered = H @ K @ H
    # Efficient eigenvalue computation: we need eigenvalues of K_centered.
    # Since we only need eigenvalues, we can use eigh on the centered matrix.
    
    # Construct Centered Kernel Matrix
    H = np.eye(N) - np.ones((N, N)) / N
    K_centered = H @ K @ H
    
    # Compute eigenvalues (ascending order by default in eigh)
    # We need the non-zero eigenvalues. K_centered is rank N-1 (at most).
    # For N=200, full eigenvalue decomposition is efficient enough.
    eigenvalues, _ = eigh(K_centered)
    
    # Sort eigenvalues in descending order and take positive ones
    # The paper mentions retaining maximum number 2m-1 (if m=n) or N-1.
    # We filter small non-zero noise.
    # eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # 6. Estimate Null Distribution
    # Asymptotic result: m * MMD^2_u -> sum_{l=1}^\infty lambda_l (z_l^2 - 2)
    # where z_l ~ N(0, 2) (i.e. sqrt(2)*N(0,1)).
    # This is equivalent to sum lambda_l * 2 * (chi^2_1 - 1)
    
    # Weights: lambda_l are eigenvalues of the integral operator.
    # Empirical estimate: lambda_hat = nu_l / N (where nu_l are eigenvalues of Gram matrix)
    # Note: In the paper Eq (5), lambda_hat = nu_l / m. 
    # However, if using the aggregate sample (size N=2m) as described in Section 4,
    # the eigenvalues scale with N. To match the variance of the MMD statistic,
    # we use weights = eigenvalues / N.
    
    weights = eigenvalues / N
    J = len(weights)
    
    if J == 0:
        return 0.0 # No signal
        
    # Generate null samples
    # z_l ~ N(0, 2) => z_l^2 - 2 ~ 2 * (chi^2_1 - 1)
    # We need sum weights * (z^2 - 2)
    
    # Efficient vectorized sampling:
    # Sample chi^2_1 (df=1)
    chi_samples = np.random.chisquare(df=1, size=(n_bootstrap, J))
    # Transform to (z^2 - 2)
    # z^2 - 2 = 2 * (chi^2_1 - 1)
    null_samples = 2 * np.sum(weights * (chi_samples - 1), axis=1)
    
    # 7. Compute Test Statistic (Scaled)
    # The statistic converges to the null distribution.
    # We scale the statistic by m (size of sample X) as per the paper's convergence theorem.
    # If m != n, we use m (primary sample size) or harmonic mean? 
    # The paper assumes m=n. Let's use m for scaling to match the theorem.
    stat_scaled = m * mmd2_u
    
    # 8. P-value
    # P-value = P(Null > Stat)
    p_value = np.mean(null_samples > stat_scaled)
    
    return p_value