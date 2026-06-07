# %%
import sys
from pathlib import Path

# Get the directory where the current script is located
script_dir = Path(__file__).parent.absolute()

# Specify the child folder name
child_folder = "DGPs"
child_path = script_dir / child_folder

# Check if the directory exists
if not child_path.exists():
    raise FileNotFoundError(f"Directory not found: {child_path}")

# Add the DGPs directory to Python's module search path
sys.path.insert(0, str(child_path))

# Import DGP modules
from dgp_alternative_set1_2 import *
from dgp_alternative_set3 import *
from dgp_alternative_set4 import *
from dgp_alternative_set5 import *
from dgp_null import *

# Import function modules
from functions import *
from functions_MMDAgg import *

import numpy as np
from joblib import Parallel, delayed


# %%
def run_mmdagg_experiment(dgp_number, mm, nn, dd, loc, scale, df, dgp_set, Nrep):
    def process_ss(ss):
        # 1. Generate Data
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
        else:
            data_dgp = dgp_choose(dgp_number, mm, nn, dd)

        Y = data_dgp[0]
        X = data_dgp[1]
        
        # 2. Run MMDAgg Test
        # B1, B2, B3 can be reduced for speed (e.g., 500), but paper defaults are high.
        # We use B1=500, B2=500 for simulation speed while maintaining reasonable accuracy.
        results = mmdagg_test(X, Y, alphas=[0.1, 0.05, 0.01], 
                              kernel="laplace_gaussian", number_bandwidths=10, 
                              B1=500, B2=500, B3=50, seed=ss)
        
        return (results[0.1], results[0.05], results[0.01])

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_ss)(ss) for ss in range(Nrep))
    rej_90, rej_95, rej_99 = zip(*results)
    
    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()


# %%
# Simulation Parameters (nn=100, mm=100 as specified)
nn = 10
mm = 100
N = nn + mm
dd_candidates = [50, 100, 500, 1000]
Nrep = 1000

# %%
print("MMDAgg Two-Sample Test Simulation")

# Set 0: Null
print("=" * 60)
print("Set 0: Null Distribution (Type I Error Calibration)")
print("=" * 60)
dgp_candidates = [1, 2, 3, 4]
for dgp in dgp_candidates:
    print(f"dgp={dgp}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mmdagg_experiment(dgp_number=dgp, mm=mm, nn=nn, dd=dd, 
                                    loc=0.0, scale=1.0, df=3, dgp_set=0, Nrep=Nrep)
        print(f"d={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 1: Location-Scale
print("=" * 60)
print("Set 1: Location-Scale Deviation (Power Study)")
print("=" * 60)

loc_scale_candidates = [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)]
for loc, scale in loc_scale_candidates:
    print(f"\\mu={loc}, \\sigma^2={scale}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mmdagg_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=loc, scale=scale, df=3, dgp_set=1, Nrep=Nrep)
        print(f"d={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 2: T-distribution
print("=" * 60)
print("Set 2: T-distribution Deviation (Power Study)")
print("=" * 60)

df_candidates = [3, 5, 10]
for df in df_candidates:
    print(f"df={df}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mmdagg_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=0.0, scale=1.0, df=df, dgp_set=3, Nrep=Nrep)
        print(f"d={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 3: Mixed Distribution
print("=" * 60)
print("Set 3: Mixed Distribution (Power Study)")
print("=" * 60)

loc_scale_candidates = [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)]
for loc, scale in loc_scale_candidates:
    print(f"\\mu={loc}, \\sigma^2={scale}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mmdagg_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=loc, scale=scale, df=3, dgp_set=4, Nrep=Nrep)
        print(f"d={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 4: Scale-Only Deviation
print("=" * 60)
print("Set 4: Scale-Only Deviation (Power Study)")
print("=" * 60)

scale_candidates = [0.6, 0.8, 1.3]
for scale in scale_candidates:
    print(f"\\sigma^2={scale}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mmdagg_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=0.0, scale=scale, df=3, dgp_set=1, Nrep=Nrep)
        print(f"d={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 5: Location-Only Deviation
print("=" * 60)
print("Set 5: Location-Only Deviation (Power Study)")
print("=" * 60)

location_candidates = [-1.0, 0.6, 1.3]
for loc in location_candidates:
    print(f"\\mu={loc}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mmdagg_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=loc, scale=1.0, df=3, dgp_set=1, Nrep=Nrep)
        print(f"d={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")