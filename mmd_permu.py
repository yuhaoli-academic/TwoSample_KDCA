# %%
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Specify the child folder name
child_folder = "DGPs"  # replace with your folder name

# Join paths and change directory
child_path = os.path.join(script_dir, child_folder)
os.chdir(child_path)


from dgp_alternative_set1_2 import *
from dgp_alternative_set3 import *
from dgp_alternative_set4 import *

from dgp_null import *


import numpy as np 
from scipy.spatial.distance import cdist,pdist
from joblib import Parallel, delayed

#%%
def mmd(dgp_number,mm,nn,dd,loc,scale,df, dgp_set, Nrep, Nb):
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
        m = X.shape[0]
        n = Y.shape[0]
        Z = np.vstack((X, Y))

        pairwise_dists = pdist(Z, 'euclidean')**2
        sigma = np.median(pairwise_dists)

        m = X.shape[0]
        n = Y.shape[0]
        
        
        K_XX = np.exp(-cdist(X, X, metric='euclidean') ** 2 / sigma)
        K_YY = np.exp(-cdist(Y, Y, metric='euclidean') ** 2 / sigma)
        K_XY = np.exp(-cdist(X, Y, metric='euclidean') ** 2 / sigma)

        k_X_non_diag = K_XX[np.triu_indices_from(K_XX, k=1)]
        k_Y_non_diag = K_YY[np.triu_indices_from(K_YY, k=1)]
        k_XY_flat = K_XY.flatten()

        
        
        # Main statistic
        stat_ker =(m+n)*( np.mean(k_X_non_diag) + np.mean(k_Y_non_diag) - 2 * np.mean(k_XY_flat)) 
        
        

        stat_kerb = np.zeros(Nb)
        for b in range(Nb):
            idx = np.random.permutation(m + n)
            Xb = Z[idx[:m], :]
            Yb = Z[idx[m:], :]

            K_XXb = np.exp(-cdist(Xb, Xb, metric='euclidean') ** 2 / sigma)
            K_YYb = np.exp(-cdist(Yb, Yb, metric='euclidean') ** 2 / sigma)
            K_XYb = np.exp(-cdist(Xb, Yb, metric='euclidean') ** 2 / sigma)

            k_X_non_diag_b = K_XXb[np.triu_indices_from(K_XXb, k=1)]
            k_Y_non_diag_b = K_YYb[np.triu_indices_from(K_YYb, k=1)]
            k_XY_flat_b = K_XYb.flatten()

            

            stat_kerb[b] = (m + n) * ( np.mean(k_X_non_diag_b) + np.mean(k_Y_non_diag_b) - 2 * np.mean(k_XY_flat_b)) 
        
        # P-value computation
        pvalue_ker = np.mean(stat_ker < stat_kerb)
        return (pvalue_ker < 0.1, pvalue_ker < 0.05, pvalue_ker < 0.01)
    
    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))
    
    # Unpack results
    rej_90, rej_95, rej_99 = zip(*results)
    
    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()

# %%

nn = 100
mm = 100
N = nn+mm
dd_candidates = [50,100, 500,1000]



#%%
print("MMD")
print("Set 0, Null Distribution")
dgp_candidates = [1, 2, 3, 4]
for dgp in dgp_candidates:
    print(f"dgp={dgp}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", mmd(dgp_number=dgp,mm=mm, nn=nn, dd=dd, loc=0.0, scale=1.0, df=3, dgp_set=0, Nrep=1000, Nb=500))
    print("\n")

# %%
print("Set 1, Location-Scale Deviation")
loc_scale_candidates = [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)]
for loc, scale in loc_scale_candidates: 
    print(f"loc={loc}, scale={scale}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", mmd(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=loc, scale=scale, df=3, dgp_set=1, Nrep=1000, Nb=500))
    print("\n")




# %%
print("Set 2, T-distribution")
df_candidates = [3, 5, 10]
for df in df_candidates:
    print(f"df={df}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", mmd(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=0.0, scale=1.0, df=df, dgp_set=3, Nrep=1000, Nb=500))
    print("\n") 

# %%
print("Set 3, Mixed Distribution")
loc_scale_candidates = [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)]
for loc, scale in loc_scale_candidates:
    print(f"loc={loc}, scale={scale}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", mmd(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=loc, scale=scale, df=3, dgp_set=4, Nrep=1000, Nb=500))
    print("\n") 

# %%
print("Set 4, Scale-Only Deviation")
scale_candidates = [0.6, 0.8, 1.3]
for scale in scale_candidates:
    print(f"scale={scale}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", mmd(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=0.0, scale=scale, df=3, dgp_set=1, Nrep=1000, Nb=500))
    print("\n")
# %%
