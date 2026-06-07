import numpy as np
import scipy.spatial

# ... [Keep your existing functions: full_matrix_gaussian, b_test_p_value, etc.] ...

def compute_pairwise_matrix_mmdagg(X, Y, l):
    Z = np.concatenate((X, Y))
    if l == "l1":
        return scipy.spatial.distance.cdist(Z, Z, 'cityblock')
    elif l == "l2":
        return scipy.spatial.distance.cdist(Z, Z, 'euclidean')
    else:
        raise ValueError("Norm type should either be 'l1' or 'l2'.")

def kernel_matrix_mmdagg(pairwise_matrix, l, kernel_type, bandwidth):
    d = pairwise_matrix / bandwidth
    if kernel_type == "gaussian" and l == "l2":
        return  np.exp(-d ** 2)
    elif kernel_type == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    elif (kernel_type == "matern_0.5_l1" and l == "l1") or (kernel_type == "matern_0.5_l2" and l == "l2") or (kernel_type == "laplace" and l == "l1"):
        return  np.exp(-d)
    elif (kernel_type == "matern_1.5_l1" and l == "l1") or (kernel_type == "matern_1.5_l2" and l == "l2"):
        return (1 + np.sqrt(3) * d) * np.exp(- np.sqrt(3) * d)
    elif (kernel_type == "matern_2.5_l1" and l == "l1") or (kernel_type == "matern_2.5_l2" and l == "l2"):
        return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(- np.sqrt(5) * d)
    elif (kernel_type == "matern_3.5_l1" and l == "l1") or (kernel_type == "matern_3.5_l2" and l == "l2"):
        return (1 + np.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * np.sqrt(7) / 3 / 5 * d ** 3) * np.exp(- np.sqrt(7) * d)
    elif (kernel_type == "matern_4.5_l1" and l == "l1") or (kernel_type == "matern_4.5_l2" and l == "l2"):
        return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * np.exp(- 3 * d)
    else:
        raise ValueError('The values of l and kernel_type are not valid.')

def create_weights_mmdagg(N, weights_type):
    if weights_type == "uniform":
        weights = np.array([1 / N,] * N)
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array([1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)])
        else:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)])
            weights = np.array([1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser) for i in range(1, N + 1)])
    else:
        raise ValueError('weights_type should be "uniform", "decreasing", "increasing", or "centred".')
    return weights

def mmdagg_test(X, Y, alphas=[0.1, 0.05, 0.01], kernel="laplace_gaussian", number_bandwidths=10, 
                weights_type="uniform", B1=500, B2=500, B3=50, seed=None):
    """
    Performs the MMDAgg test and returns results for multiple alphas.
    Returns a dictionary mapping alpha -> reject (bool).
    """
    m = X.shape[0]
    n = Y.shape[0]
    
    # Setup random state
    rs = np.random.RandomState(seed)
    
    # Bandwidths collection logic (from reference code)
    def compute_bandwidths(distances, number_bandwidths):    
        if np.min(distances) < 10 ** (-1):
            d = np.sort(distances)
            lambda_min = np.maximum(d[int(np.floor(len(d) * 0.05))], 10 ** (-1))
        else:
            lambda_min = np.min(distances)
        lambda_min = lambda_min / 2
        lambda_max = np.maximum(np.max(distances), 3 * 10 ** (-1))
        lambda_max = lambda_max * 2
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = np.array([power ** i * lambda_min for i in range(number_bandwidths)])
        return bandwidths
    
    # Compute distances for bandwidth heuristics
    max_samples = 500
    distances_l1 = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "cityblock").reshape(-1)
    bandwidths_l1 = compute_bandwidths(distances_l1, number_bandwidths)
    distances_l2 = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "euclidean").reshape(-1)
    bandwidths_l2 = compute_bandwidths(distances_l2, number_bandwidths)
    
    # Kernel list setup
    if kernel == "laplace_gaussian":
        kernel_bandwidths_l_list = [("laplace", bandwidths_l1, "l1"), ("gaussian", bandwidths_l2, "l2")]
    elif kernel == "gaussian":
        kernel_bandwidths_l_list = [("gaussian", bandwidths_l2, "l2")]
    # Add other kernel options here if needed
    else:
        kernel_bandwidths_l_list = [("laplace", bandwidths_l1, "l1"), ("gaussian", bandwidths_l2, "l2")]
        
    # Weights
    weights = create_weights_mmdagg(number_bandwidths, weights_type) / len(kernel_bandwidths_l_list)
    
    # Setup Permutations or Wild Bootstrap
    if m != n:
        approx_type = "permutations"
    else:
        approx_type = "wild bootstrap"
        
    if approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R)) 
    elif approx_type == "permutations":
        idx = rs.rand(B1 + B2 + 1, m + n).argsort(axis=1)
        v11 = np.concatenate((np.ones(m), -np.ones(n)))
        V11i = np.tile(v11, (B1 + B2 + 1, 1))
        V11 = np.take_along_axis(V11i, idx, axis=1)
        V11[B1] = v11
        V11 = V11.transpose()
        
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        V10i = np.tile(v10, (B1 + B2 + 1, 1))
        V10 = np.take_along_axis(V10i, idx, axis=1)
        V10[B1] = v10
        V10 = V10.transpose()
        
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V01i = np.tile(v01, (B1 + B2 + 1, 1))
        V01 = np.take_along_axis(V01i, idx, axis=1)
        V01[B1] = v01
        V01 = V01.transpose()

    # Compute MMD Estimates Matrix M
    N_bw = number_bandwidths * len(kernel_bandwidths_l_list)
    M = np.zeros((N_bw, B1 + B2 + 1))
    last_l_pairwise_matrix_computed = ""
    
    for j in range(len(kernel_bandwidths_l_list)):
        kern, bandwidths, l = kernel_bandwidths_l_list[j]
        if l != last_l_pairwise_matrix_computed:
            pairwise_matrix = compute_pairwise_matrix_mmdagg(X, Y, l)
            last_l_pairwise_matrix_computed = l
        for i in range(number_bandwidths):
            bw = bandwidths[i]
            K = kernel_matrix_mmdagg(pairwise_matrix, l, kern, bw)
            if approx_type == "wild bootstrap":
                np.fill_diagonal(K, 0)
                np.fill_diagonal(K[:n, n:], 0)
                np.fill_diagonal(K[n:, :n], 0)
                M[number_bandwidths * j + i] = np.sum(R * (K @ R), 0)
            elif approx_type == "permutations":
                np.fill_diagonal(K, 0)
                M[number_bandwidths * j + i] = (
                    np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                    + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                    + np.sum(V11 * (K @ V11), 0) / (m * n)
                )

    MMD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1 + 1], axis=1)
    M2 = M[:, B1 + 1:]

    # Bisection for each alpha
    results = {}
    for alpha in alphas:
        u_min = 0
        u_max = np.min(1 / weights)
        quantiles = np.zeros((N_bw, 1))
        
        for _ in range(B3):
            u = (u_max + u_min) / 2
            for j in range(len(kernel_bandwidths_l_list)):
                for i in range(number_bandwidths):
                    idx_q = number_bandwidths * j + i
                    quantiles[idx_q] = M1_sorted[idx_q, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1]
            
            P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
            if P_u <= alpha:
                u_min = u
            else:
                u_max = u
        
        # Final decision for this alpha
        u = u_min
        # Recalculate final quantiles with optimal u
        for j in range(len(kernel_bandwidths_l_list)):
            for i in range(number_bandwidths):
                idx_q = number_bandwidths * j + i
                quantiles[idx_q] = M1_sorted[idx_q, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1]
        
        reject = np.any(MMD_original > quantiles.reshape(-1))
        results[alpha] = reject
        
    return results