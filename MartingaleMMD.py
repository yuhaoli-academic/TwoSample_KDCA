import sys
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

# Setup paths
script_dir = Path(__file__).parent.absolute()
child_path = script_dir / "DGPs"
if not child_path.exists():
    raise FileNotFoundError(f"Directory not found: {child_path}")
sys.path.insert(0, str(child_path))

# Import DGPs
try:
    from dgp_alternative_set1_2 import dgp_choose_set1_2
    from dgp_alternative_set3 import dgp_choose_set3
    from dgp_alternative_set4 import dgp_choose_set4
    from dgp_null import dgp_choose
except ImportError:
    print("Warning: Could not import DGP modules.")
    def dgp_choose(*args): pass
    def dgp_choose_set1_2(*args): pass
    def dgp_choose_set3(*args): pass
    def dgp_choose_set4(*args): pass

# Import the mMMD test function
from functions_MartingaleMMD import mMMD_test

def run_mMMD_experiment(dgp_number, mm, nn, dd, loc, scale, df, dgp_set, Nrep):
    """
    Runs the mMMD simulation for a specific configuration.
    Returns p-values to allow calculation of rejection rates at multiple levels.
    """
    def process_iteration(ss):
        # 1. Generate Data
        if dgp_set == 0:
            data_dgp = dgp_choose(dgp_number, mm, nn, dd)
        elif dgp_set == 1 or dgp_set == 2:
            data_dgp = dgp_choose_set1_2(mm, nn, dd, loc, scale)
        elif dgp_set == 3:
            data_dgp = dgp_choose_set3(mm, nn, dd, df)
        elif dgp_set == 4:
            data_dgp = dgp_choose_set4(mm, nn, dd, loc, scale)
        else:
            raise ValueError(f"Unknown dgp_set: {dgp_set}")

        Y = data_dgp[0]
        X = data_dgp[1]
        
        # 2. Run mMMD Test
        _, p_val = mMMD_test(X, Y, alpha=0.05)
        
        return p_val

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_iteration)(ss) for ss in range(Nrep))
    
    # Calculate rejection rates for three levels
    p_values = np.array(results)
    rej_90 = np.mean(p_values < 0.1)
    rej_95 = np.mean(p_values < 0.05)
    rej_99 = np.mean(p_values < 0.01)
    
    return rej_90, rej_95, rej_99

# %%
nn = 100
mm = 100
N = nn + mm
dd_candidates = [50, 100, 500, 1000]
Nrep = 1000

# %%
print("mMMD Test Simulation (Martingale MMD)")

# Set 0: Null
print("=" * 60)
print("Set 0: Null Distribution (Type I Error Calibration)")
print("=" * 60)
dgp_candidates = [1, 2, 3, 4]
for dgp in dgp_candidates:
    print(f"dgp={dgp}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mMMD_experiment(dgp_number=dgp, mm=mm, nn=nn, dd=dd, 
                                    loc=0.0, scale=1.0, df=3, dgp_set=0, Nrep=Nrep)
        print(f"dd={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 1: Location-Scale
print("=" * 60)
print("Set 1: Location-Scale Deviation (Power Study)")
print("=" * 60)

loc_scale_candidates = [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)]
for loc, scale in loc_scale_candidates:
    print(f"loc={loc}, scale={scale}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mMMD_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=loc, scale=scale, df=3, dgp_set=1, Nrep=Nrep)
        print(f"dd={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
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
        r90, r95, r99 = run_mMMD_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=0.0, scale=1.0, df=df, dgp_set=3, Nrep=Nrep)
        print(f"dd={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 3: Mixed Distribution
print("=" * 60)
print("Set 3: Mixed Distribution (Power Study)")
print("=" * 60)

loc_scale_candidates = [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)]
for loc, scale in loc_scale_candidates:
    print(f"loc={loc}, scale={scale}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mMMD_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=loc, scale=scale, df=3, dgp_set=4, Nrep=Nrep)
        print(f"dd={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 4: Scale-Only Deviation
print("=" * 60)
print("Set 4: Scale-Only Deviation (Power Study)")
print("=" * 60)

scale_candidates = [0.6, 0.8, 1.3]
for scale in scale_candidates:
    print(f"scale={scale}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mMMD_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=0.0, scale=scale, df=3, dgp_set=1, Nrep=Nrep)
        print(f"dd={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")

# %%
# Set 5: Location-Only Deviation
print("=" * 60)
print("Set 5: Location-Only Deviation (Power Study)")
print("=" * 60)

location_candidates = [-1.0, 0.6, 1.3]
for loc in location_candidates:
    print(f"loc={loc}:")
    for dd in dd_candidates:
        r90, r95, r99 = run_mMMD_experiment(dgp_number=1, mm=mm, nn=nn, dd=dd, 
                                    loc=loc, scale=1.0, df=3, dgp_set=1, Nrep=Nrep)
        print(f"dd={dd}: ({r90:.3f}, {r95:.3f}, {r99:.3f})")
    print("\n")