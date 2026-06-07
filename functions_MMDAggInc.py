import numpy as np
import scipy.spatial



def create_indices_agginc(N, R):
    """
    Return lists of indices of R subdiagonals of N x N matrix.
    Corresponds to the design D = {(i, i+r) : i=1..N-r, r=1..R}.
    """
    index_X = []
    index_Y = []
    for r in range(1, R + 1):
        index_X.extend([i for i in range(N - r)])
        index_Y.extend([i + r for i in range(N - r)])
    return np.array(index_X), np.array(index_Y)

def mmdagginc_test(X, Y, alphas=[0.1, 0.05, 0.01], R=200, 
                   number_bandwidths=10, weights_type="uniform", 
                   B1=500, B2=500, B3=50, seed=None):
    """
    Efficient Aggregated MMD Test using Incomplete U-statistics (MMDAggInc).
    
    Parameters:
    X, Y: Samples.
    alphas: List of significance levels.
    R: Number of superdiagonals. If R >= min(N, M)-1, it becomes the complete quadratic test.
       Small R (e.g., 1, 10) results in linear time complexity O(N).
    """
    # Set seed
    rs = np.random.RandomState(seed)
    
    N = min(X.shape[0], Y.shape[0])
    # Truncate to balanced sample size for Incomplete U-statistic pairing
    X = X[:N]
    Y = Y[:N]
    
    # Cap R at maximum possible diagonal
    R = min(R, N - 1)
    
    # 1. Create Design Indices
    index_i, index_j = create_indices_agginc(N, R)
    size_D = len(index_i)
    if size_D == 0:
        return {a: False for a in alphas}

    # 2. Compute Bandwidths Collection
    # Heuristic from reference code
    max_samples = 500
    distances = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "euclidean").reshape(-1)
    
    if np.min(distances) < 10 ** (-1):
        dd = np.sort(distances)
        lambda_min = np.maximum(dd[int(np.floor(len(dd) * 0.05))], 10 ** (-1))
    else:
        lambda_min = np.min(distances)
    lambda_min = lambda_min / 2
    lambda_max = np.maximum(np.max(distances), 3 * 10 ** (-1))
    lambda_max = lambda_max * 2
    
    power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
    bandwidths = np.array([power ** i * lambda_min for i in range(number_bandwidths)])
    
    # 3. Compute h_values for Incomplete U-statistic
    # h_values shape: (number_bandwidths, size_D)
    # h(X_i, X_j, Y_i, Y_j) = k(X_i, X_j) - k(X_i, Y_j) - k(Y_i, X_j) + k(Y_i, Y_j)
    # Note: We use index_i and index_j for both X and Y pairs as per incomplete U-stat definition
    
    norm_Xi_Xj = np.linalg.norm(X[index_i] - X[index_j], axis=1) ** 2
    norm_Xi_Yj = np.linalg.norm(X[index_i] - Y[index_j], axis=1) ** 2
    norm_Yi_Xj = np.linalg.norm(Y[index_i] - X[index_j], axis=1) ** 2
    norm_Yi_Yj = np.linalg.norm(Y[index_i] - Y[index_j], axis=1) ** 2
    
    h_values = np.zeros((number_bandwidths, size_D))
    for r in range(number_bandwidths):
        bw = bandwidths[r]
        K_XX = np.exp(-norm_Xi_Xj / (bw ** 2))
        K_XY = np.exp(-norm_Xi_Yj / (bw ** 2))
        K_YX = np.exp(-norm_Yi_Xj / (bw ** 2))
        K_YY = np.exp(-norm_Yi_Yj / (bw ** 2))
        h_values[r] = K_XX - K_XY - K_YX + K_YY
        
    # 4. Compute Bootstrap Values
    # Generate Rademacher variables (Wild Bootstrap)
    epsilon = rs.choice([1.0, -1.0], size=(N, B1 + B2))
    
    # Compute Epsilon products for indices
    # e_values shape: (size_D, B1+B2)
    e_values = epsilon[index_i] * epsilon[index_j]
    
    # Compute statistics: M = h_values @ e_values / size_D
    # M shape: (number_bandwidths, B1+B2)
    M = np.dot(h_values, e_values) / size_D
    
    # Split into quantile estimation (B1) and p_u estimation (B2)
    # Add original statistic (epsilon=1 vector) to M1
    # Original stat corresponds to sum(h_values) / size_D
    original_stat = np.sum(h_values, axis=1) / size_D
    
    M1 = np.column_stack([M[:, :B1], original_stat.reshape(-1, 1)])
    M2 = M[:, B1:]
    
    # Sort M1 for quantile lookup
    M1_sorted = np.sort(M1, axis=1)
    
    # Weights
    if weights_type == "uniform":
        weights = np.array([1 / number_bandwidths] * number_bandwidths)
    else:
        # Add other weight types if necessary, using uniform for default
        weights = np.array([1 / number_bandwidths] * number_bandwidths)
        
    # 5. Aggregation for each alpha
    results = {}
    
    for alpha in alphas:
        # Bisection for correction u_alpha
        u_min = 0.0
        u_max = np.min(1 / weights)
        quantiles = np.zeros((number_bandwidths, 1))
        
        for _ in range(B3):
            u = (u_max + u_min) / 2
            
            # Find quantiles for current u
            # quantile index = ceil((B1 + 1) * (1 - u * w_i))
            for i in range(number_bandwidths):
                idx = int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
                idx = max(0, min(idx, B1)) # Clip index
                quantiles[i] = M1_sorted[i, idx]
            
            # Estimate P_u using M2
            # P_u = Prob( max(stat - quantile) > 0 )
            # M2 shape (K, B2), quantiles shape (K, 1)
            diffs = M2 - quantiles
            max_diffs = np.max(diffs, axis=0)
            P_u = np.mean(max_diffs > 0)
            
            if P_u <= alpha:
                u_min = u
            else:
                u_max = u
                
        # Final decision
        u = u_min
        for i in range(number_bandwidths):
            idx = int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
            idx = max(0, min(idx, B1))
            quantiles[i] = M1_sorted[i, idx]
            
        reject = np.any(original_stat > quantiles.reshape(-1))
        results[alpha] = reject
        
    return results