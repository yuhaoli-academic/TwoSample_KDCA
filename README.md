# TwoSample_KDCA

This repository contains replication code for the paper "Kernel Two Sample Testing via Directional Components Analysis".

## Dependencies

We use [UV](https://github.com/astral-sh/uv) to manage Python dependencies. To set up the environment, copy both `pyproject.toml` and `uv.lock` files and use them to synchronize package requirements.

## Code Structure

- **Main Scripts:** Python scripts in the root directory are used to reproduce the tables and figures from the main paper. The script names indicate their specific purposes.
- **DGPs:** This folder contains code for generating the datasets used in the main experiments.
- **application:** This folder includes both the data and code required to replicate the empirical studies.
- **appendix_code:** This folder contains code for the appendix exercises. For exercises related to the Mahalanobis aggregated MMD test statistics, we directly use the R code provided by Chatterjee and Bhattacharya (2025).

