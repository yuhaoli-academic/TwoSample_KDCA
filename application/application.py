# %%
import sys
from pathlib import Path


import numpy as np 
import pandas as pd
from scipy.spatial.distance import pdist

from sklearn.model_selection import train_test_split

functions_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final"

sys.path.insert(0, functions_path)

from functions import *
from functions_mmd import *
from functions_mmd_oracle import *
from functions_MahalanobisMMD import *
from functions_MMDFUSE import *

data_path = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/Code/Final/application"

sys.path.insert(0, data_path)
#%%

data_folder = Path(data_path) / 'chin'
inputs = pd.read_csv(data_folder / 'chin_inputs.csv')
outputs = pd.read_csv(data_folder / 'chin_outputs.csv')

X = inputs[outputs.values.flatten() == 1.0].reset_index(drop=True)
Y = inputs[outputs.values.flatten() == 2.0].reset_index(drop=True)

X = X.values
Y = Y.values
Z = np.vstack((X, Y))

m = X.shape[0]
n = Y.shape[0]

pairwise_dists = pdist(Z, 'euclidean')**2
sigma = np.median(pairwise_dists)

p_hat = m / (m + n)

X_train, X_test = train_test_split(X, test_size=0.5, random_state=94)
Y_train, Y_test = train_test_split(Y, test_size=0.5, random_state=94)

j = int(1)
bar_d = int(5)
Nb = 500

#%%
print("chin dataset")


print(f"p-value for TMMD(d=1) is {p_value_calculation(X, Y, j=j, sigma=sigma, p_hat=p_hat, n_bootstrap=Nb)}")

best_j = select_d(X_train, Y_train, bar_d, sigma, p_hat, n_bootstrap=Nb)
print(f"p-value for TMMD-Oracle is {p_value_calculation(X_test,Y_test,best_j,sigma,p_hat, n_bootstrap=Nb)}")

print(f"p-value for MMD is {mmd_permutation_test(X, Y, sigma, Nb=Nb)}")

print(f"p-value for MMD-Oracle is {optimized_mmd_test(X, Y, n_permutations=Nb)}")

print(f"p-value for Mahalanobis_MMD is {MMMD_test(X, Y, n_bootstrap=Nb)}")

print(f"p-value for MMD-FUSE is {mmdfuse(X, Y,  seed=None)}")

# %%

data_folder = Path(data_path) / 'khan'
inputs = pd.read_csv(data_folder / 'khan_inputs.csv')
outputs = pd.read_csv(data_folder / 'khan_outputs.csv')

X = inputs[outputs.values.flatten() == 2.0].reset_index(drop=True)
Y = inputs[outputs.values.flatten() == 4.0].reset_index(drop=True)

X = X.values
Y = Y.values
Z = np.vstack((X, Y))

m = X.shape[0]
n = Y.shape[0]

pairwise_dists = pdist(Z, 'euclidean')**2
sigma = np.median(pairwise_dists)

p_hat = m / (m + n)

X_train, X_test = train_test_split(X, test_size=0.5, random_state=94)
Y_train, Y_test = train_test_split(Y, test_size=0.5, random_state=94)

j = int(1)
bar_d = int(5)
Nb = 500

# %%
print("khan dataset")

print(f"p-value for TMMD(d=1) is {p_value_calculation(X, Y, j=j, sigma=sigma, p_hat=p_hat, n_bootstrap=Nb)}")

best_j = select_d(X_train, Y_train, bar_d, sigma, p_hat, n_bootstrap=Nb)
print(f"p-value for TMMD-Oracle is {p_value_calculation(X_test,Y_test,best_j,sigma,p_hat, n_bootstrap=Nb)}")

print(f"p-value for MMD is {mmd_permutation_test(X, Y, sigma, Nb=Nb)}")

print(f"p-value for MMD-Oracle is {optimized_mmd_test(X, Y, n_permutations=Nb)}")

print(f"p-value for Mahalanobis_MMD is {MMMD_test(X, Y, n_bootstrap=Nb)}")

print(f"p-value for MMD-FUSE is {mmdfuse(X, Y,  seed=None)}")
# %%
data_folder = Path(data_path) / 'gordon'
inputs = pd.read_csv(data_folder / 'gordon_inputs.csv')
outputs = pd.read_csv(data_folder / 'gordon_outputs.csv')

X = inputs[outputs.values.flatten() == 1.0].reset_index(drop=True)
Y = inputs[outputs.values.flatten() == 2.0].reset_index(drop=True)

X = X.values
Y = Y.values
Z = np.vstack((X, Y))

m = X.shape[0]
n = Y.shape[0]

pairwise_dists = pdist(Z, 'euclidean')**2
sigma = np.median(pairwise_dists)

p_hat = m / (m + n)

X_train, X_test = train_test_split(X, test_size=0.5, random_state=94)
Y_train, Y_test = train_test_split(Y, test_size=0.5, random_state=94)

j = int(1)
bar_d = int(5)
Nb = 500
# %%
print("gordon dataset")

print(f"p-value for TMMD(d=1) is {p_value_calculation(X, Y, j=j, sigma=sigma, p_hat=p_hat, n_bootstrap=Nb)}")

best_j = select_d(X_train, Y_train, bar_d, sigma, p_hat, n_bootstrap=Nb)
print(f"p-value for TMMD-Oracle is {p_value_calculation(X_test,Y_test,best_j,sigma,p_hat, n_bootstrap=Nb)}")

print(f"p-value for MMD is {mmd_permutation_test(X, Y, sigma, Nb=Nb)}")

print(f"p-value for MMD-Oracle is {optimized_mmd_test(X, Y, n_permutations=Nb)}")

print(f"p-value for Mahalanobis_MMD is {MMMD_test(X, Y, n_bootstrap=Nb)}")

print(f"p-value for MMD-FUSE is {mmdfuse(X, Y,  seed=None)}")
# %%
