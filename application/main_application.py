# %%
import sys
from pathlib import Path

# Get the directory where the current script is located
script_dir = Path(__file__).parent.absolute()

# Add the script's directory to Python's module search path
sys.path.insert(0, str(script_dir))




from functions import *

import numpy as np 
import pandas as pd
from scipy.spatial.distance import cdist,pdist
from scipy.sparse.linalg import eigsh
from scipy.stats import multivariate_normal
import time
#%%

data_folder = script_dir / 'chin'
inputs = pd.read_csv(data_folder / 'chin_inputs.csv')
outputs = pd.read_csv(data_folder / 'chin_outputs.csv')

X = inputs[outputs.values.flatten() == 1.0].reset_index(drop=True)
Y = inputs[outputs.values.flatten() == 2.0].reset_index(drop=True)

X = X.values
Y = Y.values
# %%
j = int(1)

#%%
print("chin dataset")

start_time = time.time()
print(f"p-value for multi-kernel is {multi(X, Y, Nb=500, j=j)}")
end_time = time.time()
print(f"Time taken for multi-kernel: {end_time - start_time} seconds")


start_time = time.time()
print(f"p-value for single-kernel is {single(X, Y, Nb=500, j=j)}")
end_time = time.time()
print(f"Time taken for single-kernel: {end_time - start_time} seconds")

start_time = time.time()
print(f"p-value for mmd is {mmd(X, Y, Nb=500)}")
end_time = time.time()
print(f"Time taken for mmd: {end_time - start_time} seconds")
# %%

data_folder = script_dir / 'khan'
inputs = pd.read_csv(data_folder / 'khan_inputs.csv')
outputs = pd.read_csv(data_folder / 'khan_outputs.csv')

X = inputs[outputs.values.flatten() == 2.0].reset_index(drop=True)
Y = inputs[outputs.values.flatten() == 4.0].reset_index(drop=True)

X = X.values
Y = Y.values
# %%
print("khan dataset")

start_time = time.time()
print(f"p-value for multi-kernel is {multi(X, Y, Nb=500, j=j)}")
end_time = time.time()
print(f"Time taken for multi-kernel: {end_time - start_time} seconds")

start_time = time.time()
print(f"p-value for single-kernel is {single(X, Y, Nb=500, j=j)}")
end_time = time.time()
print(f"Time taken for single-kernel: {end_time - start_time} seconds")

start_time = time.time()
print(f"p-value for mmd is {mmd(X, Y, Nb=500)}")
end_time = time.time()
print(f"Time taken for mmd: {end_time - start_time} seconds")
# %%
data_folder = script_dir / 'gordon'
inputs = pd.read_csv(data_folder / 'gordon_inputs.csv')
outputs = pd.read_csv(data_folder / 'gordon_outputs.csv')

X = inputs[outputs.values.flatten() == 1.0].reset_index(drop=True)
Y = inputs[outputs.values.flatten() == 2.0].reset_index(drop=True)

X = X.values
Y = Y.values
# %%
print("gordon dataset")

start_time = time.time()
print(f"p-value for multi-kernel is {multi(X, Y, Nb=500, j=j)}")
end_time = time.time()
print(f"Time taken for multi-kernel: {end_time - start_time} seconds")

start_time = time.time()
print(f"p-value for single-kernel is {single(X, Y, Nb=500, j=j)}")
end_time = time.time()
print(f"Time taken for single-kernel: {end_time - start_time} seconds")

start_time = time.time()
print(f"p-value for mmd is {mmd(X, Y, Nb=500)}")
end_time = time.time()
print(f"Time taken for mmd: {end_time - start_time} seconds")
# %%
