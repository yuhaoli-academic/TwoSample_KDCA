import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Paths and imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DGP_DIR = SCRIPT_DIR / "DGPs"

if not DGP_DIR.exists():
	raise FileNotFoundError(f"DGPs directory not found: {DGP_DIR}")

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DGP_DIR))

from dgp_alternative_set1_2 import dgp_choose_set1_2
from dgp_alternative_set3 import dgp_choose_set3
from dgp_alternative_set4 import dgp_choose_set4

from functions import p_value_calculation, select_d
from functions_bMMD import b_test_p_value
from functions_MahalanobisMMD import MMMD_test
from functions_MartingaleMMD import mMMD_test
from functions_mmd import mmd_permutation_test
from functions_mmd_oracle import optimized_mmd_test
from functions_MMDAgg import mmdagg_test
from functions_MMDAggInc import mmdagginc_test
from functions_MMDFUSE import mmdfuse
from functions_SpectralReguMMD import spectral_regularized_test
from functions_spectrum_mmd import p_value_spectrum_mmd
from functions_xmmd import xMMD_test
from functions_LinearTimeMMD import compute_linear_time_mmd

# -----------------------------------------------------------------------------
# Benchmark configuration
# -----------------------------------------------------------------------------
DD_CANDIDATES = [50, 100, 500, 1000]

SET_CONFIGS = {
	"Set1": {
		"dgp": "set1_2",
		"params": [(0.05, 0.5), (0.1, 1.3), (-0.05, 0.6)],
	},
	"Set2": {
		"dgp": "set3",
		"params": [3, 5, 10],
	},
	"Set3": {
		"dgp": "set4",
		"params": [(-0.05, 0.85), (0.0, 1.1), (0.05, 1.05)],
	},
}

METHOD_SAMPLE_SIZE = {
	"learn_multipicity_pivotal": 200,
	"mmd_oracle": 200,
}

DEFAULT_SAMPLE_SIZE = 100

# Timing repeats per (method, set, param, d) cell.
N_TIME_REPEATS = 1

# Keep computational settings aligned with the existing scripts.
N_BOOTSTRAP = 500
N_PERM_ORACLE = 500
J_SINGLE = 1
J_MULTIPLICITY = 5


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def median_sigma_sq(X, Y):
	Z = np.vstack((X, Y))
	pairwise_dists = pdist(Z, "euclidean") ** 2
	return np.median(pairwise_dists)


def generate_xy(set_name, param, mm, nn, dd):
	cfg = SET_CONFIGS[set_name]
	dgp_name = cfg["dgp"]

	if dgp_name == "set1_2":
		loc, scale = param
		Y, X = dgp_choose_set1_2(mm, nn, dd, loc, scale)
	elif dgp_name == "set3":
		df = param
		Y, X = dgp_choose_set3(mm, nn, dd, df)
	elif dgp_name == "set4":
		loc, scale = param
		Y, X = dgp_choose_set4(mm, nn, dd, loc, scale)
	else:
		raise ValueError(f"Unknown DGP name: {dgp_name}")

	return X, Y


def run_single_test(method_name, X, Y, seed):
	sigma = median_sigma_sq(X, Y)
	p_hat = X.shape[0] / (X.shape[0] + Y.shape[0])

	if method_name == "single_kernel_truncate_pivotal":
		_ = p_value_calculation(X, Y, J_SINGLE, sigma, p_hat, n_bootstrap=N_BOOTSTRAP)

	elif method_name == "learn_multipicity_pivotal":
		X_train, X_test = train_test_split(X, test_size=0.5, random_state=94)
		Y_train, Y_test = train_test_split(Y, test_size=0.5, random_state=94)
		best_j = select_d(
			X_train,
			Y_train,
			J_MULTIPLICITY,
			sigma,
			p_hat,
			n_bootstrap=N_BOOTSTRAP,
		)
		_ = p_value_calculation(
			X_test,
			Y_test,
			best_j,
			sigma,
			p_hat,
			n_bootstrap=N_BOOTSTRAP,
		)

	elif method_name == "mmd_oracle":
		_ = optimized_mmd_test(X, Y, n_permutations=N_PERM_ORACLE, seed=seed)

	elif method_name == "mmd_permu":
		_ = mmd_permutation_test(X, Y, sigma, Nb=N_BOOTSTRAP)

	elif method_name == "BMMD":
		_ = b_test_p_value(X, Y, sigma)

	elif method_name == "MahalanobisMMD":
		_ = MMMD_test(X, Y, n_bootstrap=N_BOOTSTRAP)

	elif method_name == "MartingaleMMD":
		_ = mMMD_test(X, Y, alpha=0.05)

	elif method_name == "xMMD":
		_ = xMMD_test(X, Y, alpha=0.05)

	elif method_name == "MMDFUSE":
		_ = mmdfuse(X, Y, seed=seed)

	elif method_name == "MMDAgg":
		_ = mmdagg_test(
			X,
			Y,
			alphas=[0.1, 0.05, 0.01],
			kernel="laplace_gaussian",
			number_bandwidths=10,
			B1=500,
			B2=500,
			B3=50,
			seed=seed,
		)

	elif method_name == "MMDAggInc":
		_ = mmdagginc_test(
			X,
			Y,
			alphas=[0.1, 0.05, 0.01],
			R=200,
			number_bandwidths=10,
			B1=500,
			B2=500,
			B3=50,
			seed=seed,
		)

	elif method_name == "spectrum_mmd":
		_ = p_value_spectrum_mmd(X, Y, n_bootstrap=N_BOOTSTRAP)

	elif method_name == "SpectralReguMMD":
		_ = spectral_regularized_test(X, Y, alpha=0.05, s=50, n_permutations=100)

	elif method_name == "LinearTimeMMD":
		_ = compute_linear_time_mmd(X, Y)

	else:
		raise ValueError(f"Unknown method: {method_name}")


def benchmark_method(method_name):
	n = METHOD_SAMPLE_SIZE.get(method_name, DEFAULT_SAMPLE_SIZE)
	mm = n
	nn = n
	rows = []

	for set_name, set_cfg in SET_CONFIGS.items():
		for param in set_cfg["params"]:
			for dd in DD_CANDIDATES:
				times = []
				for rep in range(N_TIME_REPEATS):
					seed = 2026 + rep
					X, Y = generate_xy(set_name, param, mm, nn, dd)
					t0 = time.perf_counter()
					run_single_test(method_name, X, Y, seed)
					elapsed = time.perf_counter() - t0
					times.append(elapsed)

				rows.append(
					{
						"method": method_name,
						"set": set_name,
						"parameter": str(param),
						"d": dd,
						"n": nn,
						"m": mm,
						"repeats": N_TIME_REPEATS,
						"time_mean_sec": float(np.mean(times)),
						"time_std_sec": float(np.std(times)),
					}
				)

	return pd.DataFrame(rows)


def summarize_timing(df):
	summary = (
		df.groupby(["method", "set", "n", "m"], as_index=False)
		.agg(
			n_cases=("time_mean_sec", "count"),
			avg_time_per_case_sec=("time_mean_sec", "mean"),
			median_time_per_case_sec=("time_mean_sec", "median"),
			total_time_sec=("time_mean_sec", "sum"),
		)
		.sort_values(["set", "avg_time_per_case_sec"])
	)
	return summary


def main():
	methods = [
		"single_kernel_truncate_pivotal",
		"learn_multipicity_pivotal",
		"mmd_oracle",
		"mmd_permu",
		"BMMD",
		"MahalanobisMMD",
		"MartingaleMMD",
		"xMMD",
		"MMDFUSE",
		"MMDAgg",
		"MMDAggInc",
		"spectrum_mmd",
		"SpectralReguMMD",
		"LinearTimeMMD",
	]

	all_detail = []
	for method_name in methods:
		print(f"Running timing for {method_name}...")
		df_method = benchmark_method(method_name)
		all_detail.append(df_method)

	detail_df = pd.concat(all_detail, ignore_index=True)
	summary_df = summarize_timing(detail_df)

	out_detail = SCRIPT_DIR / "time_comparison_detail_set1_set3.csv"
	out_summary = SCRIPT_DIR / "time_comparison_summary_set1_set3.csv"

	detail_df.to_csv(out_detail, index=False)
	summary_df.to_csv(out_summary, index=False)

	pd.set_option("display.max_rows", 200)
	pd.set_option("display.width", 200)

	print("\nTiming summary (Set1-Set3):")
	print(summary_df)
	print(f"\nSaved detailed timings to: {out_detail}")
	print(f"Saved summary timings to: {out_summary}")


if __name__ == "__main__":
	main()

