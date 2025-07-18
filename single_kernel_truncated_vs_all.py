# %%
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys 
sys.path.append('/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final/DGPs/')

from dgp_alternative_set1_2 import *
from dgp_alternative_set3 import *
from dgp_alternative_set4 import *

from dgp_null import *

import numpy as np 
from scipy.spatial.distance import cdist,pdist
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal

#%%

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
        m = X.shape[0]
        n = Y.shape[0]
        Z = np.vstack((X, Y))
        

        pairwise_dists = pdist(Z, 'euclidean')**2
        sigma = np.median(pairwise_dists)

        l2d_XX = cdist(X, X, metric='euclidean') ** 2
        

        l2d_YY = cdist(Y, Y, metric='euclidean') ** 2
        l2d_XY = cdist(X, Y, metric='euclidean') ** 2

        K_XX = np.exp(-l2d_XX / sigma)
        

        K_YY = np.exp(-l2d_YY / sigma)
        K_XY = np.exp(-l2d_XY / sigma)

        K_YX = K_XY.T
        K= np.block([[K_XX, K_XY],
                    [K_YX, K_YY]])
        
        
        eigen_val,eigen_vec = eigsh(K, k=j, which='LM')  # Get the first j eigenvectors and eigenvalues

        
        K_j = eigen_vec @ np.diag(eigen_val) @ eigen_vec.T
        
        # Select K_XX from K_j
        K_XX_from_K_j = K_j[:m, :m]
         
        # Select K_YY from K_j
        K_YY_from_K_j = K_j[m:, m:]
        
        # Select K_XY from K_j
        K_XY_from_K_j = K_j[:m, m:]
        

        k_X_non_diag = K_XX_from_K_j[np.triu_indices_from(K_XX_from_K_j, k=1)]
        k_Y_non_diag = K_YY_from_K_j[np.triu_indices_from(K_YY_from_K_j, k=1)]
        k_XY_flat = K_XY_from_K_j.flatten()        

        stat_ker =(m+n)*( np.mean(k_X_non_diag) + np.mean(k_Y_non_diag) - 2 * np.mean(k_XY_flat)) 

        
        p_hat = m / (m+n)
        # Bootstrap loop
        if m>=n:
            # Center K_XX
            C = np.eye(m) - np.ones((m, m)) / m
            K_centered = C @ K_XX_from_K_j @C
        else:
            # Center K_YY
            C = np.eye(n) - np.ones((n, n)) / n
            K_centered = C @ K_YY_from_K_j @ C
            

        
        K_centered = K_centered / m

        mu = np.zeros(K_centered.shape[0])
        Sigma = (1 / (p_hat * (1 - p_hat))) * np.eye(K_centered.shape[0])

        v_mat = multivariate_normal.rvs(mean=mu, cov=Sigma, size=Nb)

        stat_kerb=np.sum((v_mat.T * (K_centered @ v_mat.T)),axis=0) - (1 / (p_hat * (1 - p_hat))) * np.trace(K_centered)

        
        # P-value computation
        pvalue_ker = np.mean(stat_ker < stat_kerb)
        return (pvalue_ker < 0.1, pvalue_ker < 0.05, pvalue_ker < 0.01)
    
    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))
    
    # Unpack results
    rej_90, rej_95, rej_99 = zip(*results)
    
    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()


# %%

nn = 10
mm = 100
N = nn+mm
dd_candidates = [50,100, 500,1000]

jj = int(0.01*N)

# %%
print("truncate")
print("Set 0, Null Distribution")
dgp_candidates = [1, 2, 3, 4]
for dgp in dgp_candidates:
    print(f"dgp={dgp}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", truncate(dgp_number=dgp,mm=mm, nn=nn, dd=dd, loc=0.0, scale=1.0, df=3, dgp_set=0, Nrep=1000, Nb=500, j=jj))
    print("\n")
#%%
print("Set 1, Location-Scale Deviation")
loc_scale_candidates = [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)]
for loc, scale in loc_scale_candidates:
    print(f"loc={loc}, scale={scale}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=loc, scale=scale, df=3, dgp_set=1, Nrep=1000, Nb=500, j=jj))
    print("\n")




# %%
print("Set 2, T-distribution")
df_candidates = [3, 5, 10]
for df in df_candidates:
    print(f"df={df}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=0.0, scale=1.0, df=df, dgp_set=3, Nrep=1000, Nb=500, j=jj))
    print("\n") 

# %%
print("Set 3, Mixed Distribution")
loc_scale_candidates = [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)]
for loc, scale in loc_scale_candidates:
    print(f"loc={loc}, scale={scale}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=loc, scale=scale, df=3, dgp_set=4, Nrep=1000, Nb=500, j=jj))
    print("\n") 

# %%
print("Set 4, Scale-Only Deviation")
scale_candidates = [0.6, 0.8, 1.3]
for scale in scale_candidates:
    print(f"scale={scale}:")
    for dd in dd_candidates:
        print(f"dd={dd}:", truncate(dgp_number=1,mm=mm, nn=nn, dd=dd, loc=0.0, scale=scale, df=3, dgp_set=1, Nrep=1000, Nb=500, j=jj))
    print("\n")
# %%
