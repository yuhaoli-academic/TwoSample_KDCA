import numpy as np
from scipy.spatial.distance import cdist, pdist


def mmd_permutation_test(X, Y, sigma,Nb=500):
    """
    Standard MMD Two-Sample Test with Permutation.
    Uses the median heuristic for the Gaussian kernel bandwidth.
    """
    m = X.shape[0]
    n = Y.shape[0]

    

    K_XX = np.exp(-cdist(X, X, metric='euclidean') ** 2 / sigma)
    K_YY = np.exp(-cdist(Y, Y, metric='euclidean') ** 2 / sigma)
    K_XY = np.exp(-cdist(X, Y, metric='euclidean') ** 2 / sigma)
    K_YX = K_XY.T

    K = np.block([[K_XX, K_XY],
                  [K_YX, K_YY]])

    k_X_non_diag = K_XX[np.triu_indices_from(K_XX, k=1)]
    k_Y_non_diag = K_YY[np.triu_indices_from(K_YY, k=1)]
    k_XY_flat = K_XY.flatten()

    # Main statistic
    stat_ker = (m + n) * (np.mean(k_X_non_diag) + np.mean(k_Y_non_diag) - 2 * np.mean(k_XY_flat))

    stat_kerb = np.zeros(Nb)
    for b in range(Nb):
        perm = np.random.permutation(m + n)
        K_b = K[perm][:, perm]

        K_XXb = K_b[:m, :m]
        K_YYb = K_b[m:, m:]
        K_XYb = K_b[:m, m:]

        k_X_non_diag_b = K_XXb[np.triu_indices_from(K_XXb, k=1)]
        k_Y_non_diag_b = K_YYb[np.triu_indices_from(K_YYb, k=1)]
        k_XY_flat_b = K_XYb.flatten()

        stat_kerb[b] = (m + n) * (np.mean(k_X_non_diag_b) + np.mean(k_Y_non_diag_b) - 2 * np.mean(k_XY_flat_b))

    # P-value computation
    pvalue_ker = np.mean(stat_ker < stat_kerb)

    
    return pvalue_ker