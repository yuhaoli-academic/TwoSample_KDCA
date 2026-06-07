# %%
import sys

# Define direct paths
dgp_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final/DGPs"
functions_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final"

# Add the directories to Python's module search path
sys.path.insert(0, dgp_path)
sys.path.insert(0, functions_path)

# Now import the modules
from dgp_alternative_set1_2 import *
from dgp_alternative_set3 import *
from dgp_alternative_set4 import *
from dgp_null import *
from functions import *

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# %%
d_bar = 50
loc = 0.6
scale = 1.0
mm = 100
nn = 100
dd  = 100

def compute_eigenvalues(mm, nn, dd, loc, scale):
    """Run one repetition: generate data, compute kernel, return top d_bar eigenvalues."""
    data_dgp = dgp_choose_set1_2(mm, nn, dd, loc, scale)

    Y = data_dgp[0]
    X = data_dgp[1]
    N = mm + nn
    Z = np.vstack((X, Y))

    pairwise_dists = pdist(Z, 'euclidean')**2
    sigma = np.median(pairwise_dists)

    K = full_matrix_gaussian(X, Y, sigma) / N
    eigen_val, eigen_vec = eigsh(K, k=d_bar, which='LM')
    idx = np.argsort(eigen_val)[::-1]
    eigen_val = eigen_val[idx]

    return eigen_val

# %%
n_reps = 1000

# Run in parallel
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(compute_eigenvalues)(mm, nn, dd, loc, scale)
    for _ in range(n_reps)
)

# Stack results: shape (n_reps, d_bar)
eigenvalues_all = np.stack(results, axis=0)

# %%
# Compute sample mean
eigen_mean = np.mean(eigenvalues_all, axis=0)

# %%
# Plot
ranks = np.arange(1, d_bar + 1)

plt.figure(figsize=(12, 6))
plt.plot(ranks, eigen_mean, 'b-o', markersize=4, label='Sample Mean')
plt.xlabel('Eigenvalue Rank', fontsize=13)
plt.ylabel('Eigenvalue', fontsize=13)
plt.title('Sample Average of Top 50 Eigenvalues', fontsize=14)
plt.xticks(ranks)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Export as PDF to the specified path
save_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/draft/arxiv_v2/art/eigen_value_loc_only/eigen_value_loc_only.pdf"
plt.savefig(save_path)

plt.show()
# %%