# TwoSample_KDCA

This repository contains replication code for the paper "[Kernel Two Sample Testing via Directional Components Analysis](https://arxiv.org/abs/2508.08564)".

## Dependencies

We use [pixi](https://pixi.prefix.dev/latest/) to manage Python dependencies. To set up the environment, copy both `pixi.toml` and `pixi.lock` files and use them to synchronize package requirements.

## Code Structure

- **Root experiment scripts:** Method-specific scripts in the root directory reproduce the main simulation results, including `BMMD.py`, `LinearTimeMMD.py`, `MMDAgg_run_file.py`, `MMDAggInc_run_file.py`, `MMDFUSE.py`, `MahalanobisMMD.py`, `MartingaleMMD.py`, `SpectralReguMMD.py`, `spectrum_mmd_test.py`, `mmd_oracle.py`, `mmd_permu.py`, `xMMD.py`, `time_comparison.py`, `learning_mb.py`, `learn_multipicity_pivotal.py`, and `single_kernel_truncate_pivotal.py`.
- **Root utility modules:** Shared helper functions are implemented in files such as `functions.py`, `functions_bMMD.py`, `functions_MMDAgg.py`, `functions_MMDAggInc.py`, `functions_MMDFUSE.py`, `functions_MahalanobisMMD.py`, `functions_MartingaleMMD.py`, `functions_SpectralReguMMD.py`, `functions_LinearTimeMMD.py`, `functions_mmd.py`, `functions_mmd_oracle.py`, `functions_spectrum_mmd.py`, and `functions_xmmd.py`.
- **`DGPs/`:** Data-generating process code for null and alternative settings (`dgp_null.py`, `dgp_alternative_set1_2.py`, `dgp_alternative_set3.py`, `dgp_alternative_set4.py`, `dgp_alternative_set5.py`).
- **`application/`:** Empirical application pipeline and data. This folder contains `main_application.py`/`application.py`, helper code (`functions.py`), and datasets for the `chin`, `gordon`, and `khan` studies.
- **`appendix_code/`:** Appendix experiments organized by topic: `choice_d_bar/`, `freq_d/`, `loc_only_eigenvalue/`, and `multi_kernel/`.
- **`results/`:** Saved outputs from simulation and benchmarking runs (method-specific `.txt` result files and timing summaries in `.csv`), with additional splits under `balanced/`, `unbalanced/`, and `insensitivity/`.

