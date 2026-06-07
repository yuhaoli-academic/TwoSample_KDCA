# %%
import sys

from collections import Counter

functions_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final/appendix_code/multi_kernel"

sys.path.insert(0, functions_path)

from functions_multi import *


data_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final/DGPs"

sys.path.insert(0, data_path)

# Now import the modules
from dgp_alternative_set1_2 import *
from dgp_alternative_set3 import *
from dgp_alternative_set4 import *
from dgp_null import *


import numpy as np
from scipy.spatial.distance import cdist,pdist
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split

# %%
def truncate(dgp_number,mm,nn,dd,loc,scale,df, dgp_set, Nrep, Nb,j):
    def process_ss(ss):
        if dgp_set == 0:
            data_dgp = dgp_choose(dgp_number, mm,nn,dd)
        elif dgp_set == 1:
            data_dgp = dgp_choose_set1_2(mm,nn,dd,loc,scale)
        elif dgp_set == 2:
            data_dgp = dgp_choose_set1_2(mm,nn,dd,loc,scale)
        elif dgp_set == 3:
            data_dgp = dgp_choose_set3(mm,nn,dd,df)
        elif dgp_set == 4:
            data_dgp = dgp_choose_set4(mm,nn,dd,loc,scale)

        Y = data_dgp[0]
        X = data_dgp[1]
        p_hat = X.shape[0] / (X.shape[0] + Y.shape[0])
        Z = np.vstack((X, Y))

        # Median heuristic for sigma (computed once on pooled data)
        pairwise_dists = pdist(Z, 'euclidean')**2
        sigma = np.median(pairwise_dists)
        
        # Define identical sigma list for the 3 kernels
        sigma_list = [sigma, sigma, sigma]
        
        # Define the multiple kernel types to combine
        kernel_type_list = ['gaussian', 'laplacian', 'imq']

        X_train, X_test = train_test_split(X, test_size=0.5, random_state=94)
        Y_train, Y_test = train_test_split(Y, test_size=0.5, random_state=94)

        # Data-driven selection of d for each kernel independently
        best_d_list = select_d_ratio_joint(X_train, Y_train, j, sigma_list, p_hat, kernel_type_list)

        # Compute the joint p-value using the selected dimensions, sigmas, and kernel types
        pvalue_ker = compute_joint_p_value(X_test, Y_test, best_d_list, sigma_list, p_hat, n_bootstrap=Nb, kernel_type_list=kernel_type_list)
        
        # Return best_d_list along with rejection indicators
        return (pvalue_ker < 0.1, pvalue_ker < 0.05, pvalue_ker < 0.01, tuple(best_d_list))

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))

    # Unpack results
    rej_90, rej_95, rej_99, best_ds = zip(*results)

    # Calculate the frequency of each combination of d_list
    freq_d = Counter(best_ds)

    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item(), freq_d

# %%

nn = 200
mm = 200
N = nn+mm
dd_candidates = [50,100, 500,1000]

jj = int(5)

# Dictionary to store results so we don't have to run the simulation twice
results_storage = {}

# %%
print("truncate")
print("Set 0, Null Distribution")
dgp_candidates = [1, 2, 3, 4]
for dgp in dgp_candidates:
    print(f"dgp={dgp}:")
    results_storage[f"Set0_dgp{dgp}"] = {}
    for dd in dd_candidates:
        rej_90, rej_95, rej_99, freq_d = truncate(dgp_number=dgp,mm=mm, nn=nn, dd=dd, loc=0.0, scale=1.0, df=3, dgp_set=0, Nrep=1000, Nb=500, j=jj)
        results_storage[f"Set0_dgp{dgp}"][dd] = (rej_90, rej_95, rej_99, freq_d)
        print(f"d={dd}:", (rej_90, rej_95, rej_99))
    print("\n")

#%%
print("Set 1, Location-Scale Deviation")
loc_scale_candidates = [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)]
for loc, scale in loc_scale_candidates:
    print(f"\\mu={loc}, \\sigma^2={scale}:")
    results_storage[f"Set1_loc{loc}_scale{scale}"] = {}
    for dd in dd_candidates:
        rej_90, rej_95, rej_99, freq_d = truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=loc, scale=scale, df=3, dgp_set=1, Nrep=1000, Nb=500, j=jj)
        results_storage[f"Set1_loc{loc}_scale{scale}"][dd] = (rej_90, rej_95, rej_99, freq_d)
        print(f"d={dd}:", (rej_90, rej_95, rej_99))
    print("\n")

# %%
print("Set 2, T-distribution")
df_candidates = [3, 5, 10]
for df in df_candidates:
    print(f"df={df}:")
    results_storage[f"Set2_df{df}"] = {}
    for dd in dd_candidates:
        rej_90, rej_95, rej_99, freq_d = truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=0.0, scale=1.0, df=df, dgp_set=3, Nrep=1000, Nb=500, j=jj)
        results_storage[f"Set2_df{df}"][dd] = (rej_90, rej_95, rej_99, freq_d)
        print(f"d={dd}:", (rej_90, rej_95, rej_99))
    print("\n")

# %%
print("Set 3, Mixed Distribution")
loc_scale_candidates = [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)]
for loc, scale in loc_scale_candidates:
    print(f"\\mu={loc}, \\sigma^2={scale}:")
    results_storage[f"Set3_loc{loc}_scale{scale}"] = {}
    for dd in dd_candidates:
        rej_90, rej_95, rej_99, freq_d = truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=loc, scale=scale, df=3, dgp_set=4, Nrep=1000, Nb=500, j=jj)
        results_storage[f"Set3_loc{loc}_scale{scale}"][dd] = (rej_90, rej_95, rej_99, freq_d)
        print(f"d={dd}:", (rej_90, rej_95, rej_99))
    print("\n")

# %%
print("Set 4, Scale-Only Deviation")
scale_candidates = [0.6, 0.8, 1.3]
for scale in scale_candidates:
    print(f"\\sigma^2={scale}:")
    results_storage[f"Set4_scale{scale}"] = {}
    for dd in dd_candidates:
        rej_90, rej_95, rej_99, freq_d = truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=0.0, scale=scale, df=3, dgp_set=1, Nrep=1000, Nb=500, j=jj)
        results_storage[f"Set4_scale{scale}"][dd] = (rej_90, rej_95, rej_99, freq_d)
        print(f"d={dd}:", (rej_90, rej_95, rej_99))
    print("\n")

# %%
print("Set 5, Location-Only Deviation")
location_candidates = [-1.0, 0.6, 1.3]
for loc in location_candidates:
    print(f"\\mu={loc}:")
    results_storage[f"Set5_loc{loc}"] = {}
    for dd in dd_candidates:
        rej_90, rej_95, rej_99, freq_d = truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=loc, scale=1.0, df=3, dgp_set=1, Nrep=1000, Nb=500, j=jj)
        results_storage[f"Set5_loc{loc}"][dd] = (rej_90, rej_95, rej_99, freq_d)
        print(f"d={dd}:", (rej_90, rej_95, rej_99))
    print("\n")


