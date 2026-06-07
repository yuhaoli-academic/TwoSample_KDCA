# %%
import sys
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

# Import DGPs and the new function
# Ensure functions_spectrum_mmd.py is in the same directory or path
script_dir = Path(__file__).parent.absolute()
child_path = script_dir / "DGPs"
if not child_path.exists():
    raise FileNotFoundError(f"Directory not found: {child_path}")
sys.path.insert(0, str(child_path))

# Import DGP functions (assuming structure from previous file)
# We wrap imports in try-except to handle cases where DGP files might differ
try:
    from dgp_alternative_set1_2 import dgp_choose_set1_2
    from dgp_alternative_set3 import dgp_choose_set3
    from dgp_alternative_set4 import dgp_choose_set4
    from dgp_null import dgp_choose
except ImportError:
    print("Warning: Could not import all DGP modules. Please check DGPs folder.")
    # Define dummies if imports fail to allow script parsing
    def dgp_choose(*args): pass
    def dgp_choose_set1_2(*args): pass
    def dgp_choose_set3(*args): pass
    def dgp_choose_set4(*args): pass

from functions_spectrum_mmd import p_value_spectrum_mmd
# %%
def run_spectrum_mmd_test(dgp_number, mm, nn, dd, loc, scale, df, dgp_set, Nrep):
    """
    Runs the spectrum MMD test simulation.
    """
    def process_iteration(ss):
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
            raise ValueError("Unknown dgp_set")

        Y = data_dgp[0]
        X = data_dgp[1]
        
        # 2. Run Spectrum MMD Test
        # No train/test split needed as we use median heuristic and full spectrum (Spec algorithm)
        p_val = p_value_spectrum_mmd(X, Y, n_bootstrap=500)
        
        # 3. Check rejection
        return (p_val < 0.1, p_val < 0.05, p_val < 0.01)

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_iteration)(ss) for ss in range(Nrep))
    
    # Unpack results
    rej_90, rej_95, rej_99 = zip(*results)
    
    return np.mean(rej_90).item(), np.mean(rej_95).item(), np.mean(rej_99).item()

# %%
nn = 100
mm = 100
dd_candidates = [50, 100, 500, 1000]
Nrep = 1000 # Repetitions
# %%
print("Spectrum MMD Test Simulation")

print("=" * 60)
print("Set 0: Null Distribution (Type I Error Calibration)")
print("=" * 60)
dgp_candidates = [1, 2, 3, 4]
for dgp in dgp_candidates:
    print(f"\ndgp={dgp}:")
    for dd in dd_candidates:
        rej_rates = run_spectrum_mmd_test(
            dgp_number=dgp, mm=mm, nn=nn, dd=dd,
            loc=0.0, scale=1.0, df=3, dgp_set=0,
            Nrep=Nrep
        )
        print(f"d={dd}: ({rej_rates[0]:.3f}, {rej_rates[1]:.3f}, {rej_rates[2]:.3f})")
print("\n")

# %%
print("=" * 60)
print("Set 1: Location-Scale Deviation (Power Study)")
print("=" * 60)
loc_scale_candidates = [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)]
for loc, scale in loc_scale_candidates:
    print(f"\n\\mu={loc}, \\sigma^2={scale}:")
    for dd in dd_candidates:
        rej_rates = run_spectrum_mmd_test(
            dgp_number=1, mm=mm, nn=nn, dd=dd,
            loc=loc, scale=scale, df=3, dgp_set=1,
            Nrep=Nrep
        )
        print(f"d={dd}: ({rej_rates[0]:.3f}, {rej_rates[1]:.3f}, {rej_rates[2]:.3f})")
print("\n")

# %%
print("=" * 60)
print("Set 2: T-distribution Deviation (Power Study)")
print("=" * 60)
df_candidates = [3, 5, 10]
for df in df_candidates:
    print(f"\ndf={df}:")
    for dd in dd_candidates:
        rej_rates = run_spectrum_mmd_test(
            dgp_number=1, mm=mm, nn=nn, dd=dd,
            loc=0.0, scale=1.0, df=df, dgp_set=3,
            Nrep=Nrep
        )
        print(f"d={dd}: ({rej_rates[0]:.3f}, {rej_rates[1]:.3f}, {rej_rates[2]:.3f})")
print("\n")

# %%
print("=" * 60)
print("Set 3: Mixed Distribution (Power Study)")
print("=" * 60)
loc_scale_candidates = [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)]
for loc, scale in loc_scale_candidates:
    print(f"\n\\mu={loc}, \\sigma^2={scale}:")
    for dd in dd_candidates:
        rej_rates = run_spectrum_mmd_test(
            dgp_number=1, mm=mm, nn=nn, dd=dd,
            loc=loc, scale=scale, df=3, dgp_set=4,
            Nrep=Nrep
        )
        print(f"d={dd}: ({rej_rates[0]:.3f}, {rej_rates[1]:.3f}, {rej_rates[2]:.3f})")
print("\n")

# %%
print("=" * 60)
print("Set 4: Scale-Only Deviation (Power Study)")
print("=" * 60)
scale_candidates = [0.6, 0.8, 1.3]
for scale in scale_candidates:
    print(f"\n\\sigma^2={scale}:")
    for dd in dd_candidates:
        rej_rates = run_spectrum_mmd_test(
            dgp_number=1, mm=mm, nn=nn, dd=dd,
            loc=0.0, scale=scale, df=3, dgp_set=1,
            Nrep=Nrep
        )
        print(f"d={dd}: ({rej_rates[0]:.3f}, {rej_rates[1]:.3f}, {rej_rates[2]:.3f})")
print("\n")

# %%
print("=" * 60)
print("Set 5: Location-Only Deviation (Power Study)")
print("=" * 60)
location_candidates = [-1.0, 0.6, 1.3]
for loc in location_candidates:
    print(f"\n\\mu={loc}:")
    for dd in dd_candidates:
        rej_rates = run_spectrum_mmd_test(
            dgp_number=1, mm=mm, nn=nn, dd=dd,
            loc=loc, scale=1.0, df=3, dgp_set=1,
            Nrep=Nrep
        )
        print(f"d={dd}: ({rej_rates[0]:.3f}, {rej_rates[1]:.3f}, {rej_rates[2]:.3f})")
print("\n")

    
# %%
