# %%
import sys


functions_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final/appendix_code/multi_kernel"

sys.path.insert(0, functions_path)

from functions_multi import compute_joint_p_value

data_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final/DGPs"

sys.path.insert(0, data_path)

# Now import the DGP modules
from dgp_alternative_set1_2 import *
from dgp_alternative_set3 import *
from dgp_alternative_set4 import *
from dgp_null import *



import numpy as np
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed

# %%

def truncate(dgp_number, mm, nn, dd, loc, scale, df, dgp_set, Nrep, Nb, j):
    """
    Run the multiple kernel test for a given DGP configuration and return rejection rates.
    Uses a predetermined truncation dimension j for all kernels.
    """
    def process_ss(ss):
        # Generate data according to the specified DGP set
        if dgp_set == 0:
            data_dgp = dgp_choose(dgp_number, mm, nn, dd)
        elif dgp_set == 1:
            data_dgp = dgp_choose_set1_2(mm, nn, dd, loc, scale)
        elif dgp_set == 2:
            data_dgp = dgp_choose_set1_2(mm, nn, dd, loc, scale)
        elif dgp_set == 3:
            data_dgp = dgp_choose_set3(mm, nn, dd, df)
        elif dgp_set == 4:
            data_dgp = dgp_choose_set4(mm, nn, dd, loc, scale)

        Y = data_dgp[0]
        X = data_dgp[1]
        m = X.shape[0]
        n = Y.shape[0]
        Z = np.vstack((X, Y))

        # Compute sigma as median of squared pairwise distances
        pairwise_dists = pdist(Z, 'euclidean') ** 2
        sigma = np.median(pairwise_dists)

        # Proportion parameter for asymptotic covariance
        p_hat = m / (m + n)

        # Define identical sigma list for the 3 kernels
        sigma_list = [sigma, sigma, sigma]
        
        # Define the multiple kernel types to combine
        kernel_type_list = ['gaussian', 'laplacian', 'imq']
        
        # Use the predetermined truncation dimension j for all kernels
        d_list = [j, j, j]

        # Compute joint p-value using the optimized function
        pvalue_ker = compute_joint_p_value(X, Y, d_list, sigma_list, p_hat, Nb, kernel_type_list)

        # Return rejection indicators for significance levels 0.10, 0.05, 0.01
        return (pvalue_ker < 0.10, pvalue_ker < 0.05, pvalue_ker < 0.01)

    # Parallel processing over replications
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))

    # Unpack results
    rej_90, rej_95, rej_99 = zip(*results)

    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()


# %%

nn = 100
mm = 100
N = nn + mm
dd_candidates = [50, 100, 500, 1000]

jj = int(1)

# %%
print("truncate (Multiple Kernel: Gaussian + Laplacian + IMQ)")
print("Set 0, Null Distribution")
dgp_candidates = [1, 2, 3, 4]
for dgp in dgp_candidates:
    print(f"dgp={dgp}:")
    for dd in dd_candidates:
        print(f"d={dd}:", truncate(dgp_number=dgp, mm=mm, nn=nn, dd=dd,
                                   loc=0.0, scale=1.0, df=3, dgp_set=0,
                                   Nrep=1000, Nb=500, j=jj))
    print("\n")

# %%
print("Set 1, Location-Scale Deviation")
loc_scale_candidates = [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)]
for loc, scale in loc_scale_candidates:
    print(f"\\mu={loc}, \\sigma^2={scale}:")
    for dd in dd_candidates:
        print(f"d={dd}:", truncate(dgp_number=1, mm=mm, nn=nn, dd=dd,
                                   loc=loc, scale=scale, df=3, dgp_set=1,
                                   Nrep=1000, Nb=500, j=jj))
    print("\n")

# %%
print("Set 2, T-distribution")
df_candidates = [3, 5, 10]
for df in df_candidates:
    print(f"df={df}:")
    for dd in dd_candidates:
        print(f"d={dd}:", truncate(dgp_number=1, mm=mm, nn=nn, dd=dd,
                                   loc=0.0, scale=1.0, df=df, dgp_set=3,
                                   Nrep=1000, Nb=500, j=jj))
    print("\n")

# %%
print("Set 3, Mixed Distribution")
loc_scale_candidates = [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)]
for loc, scale in loc_scale_candidates:
    print(f"\\mu={loc}, \\sigma^2={scale}:")
    for dd in dd_candidates:
        print(f"d={dd}:", truncate(dgp_number=1, mm=mm, nn=nn, dd=dd,
                                   loc=loc, scale=scale, df=3, dgp_set=4,
                                   Nrep=1000, Nb=500, j=jj))
    print("\n")

# %%
print("Set 4, Scale-Only Deviation")
scale_candidates = [0.6, 0.8, 1.3]
for scale in scale_candidates:
    print(f"\\sigma^2={scale}:")
    for dd in dd_candidates:
        print(f"d={dd}:", truncate(dgp_number=1, mm=mm, nn=nn, dd=dd,
                                   loc=0.0, scale=scale, df=3, dgp_set=1,
                                   Nrep=1000, Nb=500, j=jj))
    print("\n")

# %%
print("Set 5, Location-Only Deviation")
location_candidates = [-1.0, 0.6, 1.3]
for loc in location_candidates:
    print(f"\\mu={loc}:")
    for dd in dd_candidates:
        print(f"d={dd}:", truncate(dgp_number=1, mm=mm, nn=nn, dd=dd,
                                   loc=loc, scale=1.0, df=3, dgp_set=1,
                                   Nrep=1000, Nb=500, j=jj))
    print("\n")