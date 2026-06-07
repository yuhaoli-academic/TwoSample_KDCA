#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# Configuration
# ============================================================================
DATA_FILE = "results_for_plotting_uniform.txt"
SAVE_DIR = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/draft/arxiv_v2/art/balanced/"

# All test statistics that appear in the file (updated based on the cleaned file)
METHODS = [
    'BMMD',
    'xMMD',
    'MahalanobisMMD',
    'MartingaleMMD',
    'MMD',
    'MMD-Oracle',
    'MMDAgg',
    'MMDAggInc',
    'MMDFUSE',
    'SpectraRegu',
    'SpectrumMMD',
    'TMMD(d=1)',
    'TMMD-Oracle'
]

# Plotting styles for each method
METHOD_LABELS = {
    'BMMD': 'BMMD',
    'xMMD': 'xMMD',
    'MahalanobisMMD': 'Mahalanobis-MMD',
    'MartingaleMMD': 'Martingale-MMD',
    'MMD': 'MMD',
    'MMD-Oracle': 'MMD-Oracle',
    'MMDAgg': 'MMDAgg',
    'MMDAggInc': 'MMDAggInc',
    'MMDFUSE': 'MMD-FUSE',
    'SpectraRegu': 'SpectraRegu',
    'SpectrumMMD': 'Spectrum-MMD',
    'TMMD(d=1)': 'TMMD(d=1)',
    'TMMD-Oracle': 'TMMD-Oracle'
}

METHOD_STYLES = {
    'BMMD': {'marker': 'x', 'color': '#1f77b4', 'linestyle': '--'},
    'xMMD': {'marker': '+', 'color': '#17becf', 'linestyle': '--'},
    'MahalanobisMMD': {'marker': 'o', 'color': '#ff7f0e', 'linestyle': '-'},
    'MartingaleMMD': {'marker': 's', 'color': '#2ca02c', 'linestyle': '-'},
    'MMD': {'marker': '^', 'color': '#d62728', 'linestyle': '-'},
    'MMD-Oracle': {'marker': 'v', 'color': '#e377c2', 'linestyle': '--'},
    'MMDAgg': {'marker': 'D', 'color': '#9467bd', 'linestyle': '--'},
    'MMDAggInc': {'marker': 'v', 'color': '#8c564b', 'linestyle': '--'},
    'MMDFUSE': {'marker': 'p', 'color': '#e377c2', 'linestyle': '-.'},
    'SpectraRegu': {'marker': 'h', 'color': '#7f7f7f', 'linestyle': '-.'},
    'SpectrumMMD': {'marker': '<', 'color': '#bcbd22', 'linestyle': '-.'},
    'TMMD(d=1)': {'marker': '*', 'color': '#bcbd22', 'linestyle': ':'},
    'TMMD-Oracle': {'marker': '>', 'color': '#17becf', 'linestyle': ':'}
}

DIMENSIONS = [50, 100, 500, 1000]

# Parameter sets in canonical form (the keys we will use after normalization)
SET_PARAMS = {
    'Set 0': ['dgp=1', 'dgp=2', 'dgp=3', 'dgp=4'],
    'Set 1': [r'\mu=0.05, \sigma^2=0.5', r'\mu=0.1, \sigma^2=1.3', r'\mu=-0.05, \sigma^2=0.6'],
    'Set 2': ['df=3', 'df=5', 'df=10'],
    'Set 3': [r'\mu=-0.05, \sigma^2=0.85', r'\mu=0.0, \sigma^2=1.1', r'\mu=0.05, \sigma^2=1.05'],
    'Set 4': [r'\sigma^2=0.6', r'\sigma^2=0.8', r'\sigma^2=1.3'],
    'Set 5': [r'\mu=-1.0', r'\mu=0.6', r'\mu=1.3']
}

# Pretty titles for subplots
PARAM_TITLES = {
    'dgp=1': 'DGP 1', 'dgp=2': 'DGP 2', 'dgp=3': 'DGP 3', 'dgp=4': 'DGP 4',
    r'\mu=0.05, \sigma^2=0.5': r'$\mu=0.05,\ \sigma^2=0.5$',
    r'\mu=0.1, \sigma^2=1.3': r'$\mu=0.1,\ \sigma^2=1.3$',
    r'\mu=-0.05, \sigma^2=0.6': r'$\mu=-0.05,\ \sigma^2=0.6$',
    'df=3': r'$df=3$', 'df=5': r'$df=5$', 'df=10': r'$df=10$',
    r'\mu=-0.05, \sigma^2=0.85': r'$\mu=-0.05,\ \sigma^2=0.85$',
    r'\mu=0.0, \sigma^2=1.1': r'$\mu=0.0,\ \sigma^2=1.1$',
    r'\mu=0.05, \sigma^2=1.05': r'$\mu=0.05,\ \sigma^2=1.05$',
    r'\sigma^2=0.6': r'$\sigma^2=0.6$',
    r'\sigma^2=0.8': r'$\sigma^2=0.8$',
    r'\sigma^2=1.3': r'$\sigma^2=1.3$',
    r'\mu=-1.0': r'$\mu=-1.0$',
    r'\mu=0.6': r'$\mu=0.6$',
    r'\mu=1.3': r'$\mu=1.3$'
}

SET_TITLES = {
    'Set 0': 'Null Distribution',
    'Set 1': 'Location-Scale Deviation',
    'Set 2': 't-Distribution Deviation',
    'Set 3': 'Mixed Distribution',
    'Set 4': 'Scale-Only Deviation',
    'Set 5': 'Location-Only Deviation'
}


# ============================================================================
# Parameter canonicalization
# ============================================================================
def canonicalize_param(param_str):
    """
    Convert various parameter string formats into a single canonical form.
    Examples:
        '\mu=0.05, \sigma^20.5'   -> '\mu=0.05, \sigma^2=0.5'
        '\mu=0.05, \sigma^2=0.5'  -> '\mu=0.05, \sigma^2=0.5'
        '\sigma^20.6'             -> '\sigma^2=0.6'
        '\sigma^2=0.6'            -> '\sigma^2=0.6'
        'df=3'                    -> 'df=3'
        'dgp=1'                   -> 'dgp=1'
    """
    param_str = param_str.strip()
    # Replace Unicode micro (µ) with LaTeX \mu
    param_str = param_str.replace('µ', r'\mu')
    # Fix missing equals after \sigma^2 (e.g., '\sigma^20.5' -> '\sigma^2=0.5')
    param_str = re.sub(r'(\\sigma\^2)(\d)', r'\1=\2', param_str)
    # Ensure equals sign exists for \mu and \sigma^2 (just in case)
    param_str = re.sub(r'(\\mu)\s+(\d)', r'\1=\2', param_str)
    # Normalize spaces around commas and equals
    param_str = re.sub(r'\s*,\s*', ', ', param_str)
    param_str = re.sub(r'\s*=\s*', '=', param_str)
    return param_str


# ============================================================================
# Data parser
# ============================================================================
def parse_results(filepath):
    """
    Parse the uniform data file where each data line is: d=XX: (value)
    Returns a nested dict: data[method][set][param][dim] = value
    Parameter strings are canonicalized.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = {}
    current_method = None
    current_set = None
    current_param = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect method header (exact match)
        if stripped in METHODS:
            current_method = stripped
            data[current_method] = {}
            current_set = None
            current_param = None
            continue

        # Detect set header (e.g., "Set 0: Null Distribution ...")
        set_match = re.match(r'Set\s*([0-5])[,:]', stripped, re.IGNORECASE)
        if set_match:
            current_set = f"Set {set_match.group(1)}"
            if current_method:
                data[current_method][current_set] = {}
            current_param = None
            continue

        # Detect parameter line (starts with dgp=, df=, \mu, or \sigma)
        param_line = None
        if stripped.startswith('dgp=') or stripped.startswith('df='):
            param_line = stripped.split(':')[0].strip()
        elif '\\mu' in stripped or 'µ' in stripped or '¥mu' in stripped:
            param_line = stripped.split(':')[0].strip()
        elif '\\sigma' in stripped or '¥sigma' in stripped:
            param_line = stripped.split(':')[0].strip()

        if param_line and current_method and current_set:
            current_param = canonicalize_param(param_line)
            # Debug output (can be commented later)
            print(f"DEBUG: Method={current_method}, Set={current_set}, Param={current_param}")
            if current_param not in data[current_method][current_set]:
                data[current_method][current_set][current_param] = {}
            continue

        # Parse data lines: d=50: (0.063)
        if current_method and current_set and current_param:
            m = re.match(r'd\s*=\s*(\d+)\s*:\s*\(\s*([\d.]+)\s*\)', stripped)
            if m:
                dim = int(m.group(1))
                value = float(m.group(2))
                data[current_method][current_set][current_param][dim] = value

    return data


# ============================================================================
# Plotting functions
# ============================================================================
def plot_parameter(ax, data, method, set_name, param, dims):
    """Plot one method's rejection rates for a given set and parameter."""
    if method not in data:
        return
    if set_name not in data[method]:
        return
    if param not in data[method][set_name]:
        return
    param_dict = data[method][set_name][param]
    rates = [param_dict.get(d, None) for d in dims]
    if any(r is not None for r in rates):
        style = METHOD_STYLES.get(method, {'marker': 'o', 'color': 'black', 'linestyle': '-'})
        ax.plot(dims, rates,
                marker=style['marker'],
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=1.5,
                markersize=7,
                label=METHOD_LABELS.get(method, method))


def create_figure_with_grid(set_name, params, data, dims, ncols):
    """Create a figure with subplots arranged in a grid (ncols columns)."""
    n_params = len(params)
    nrows = (n_params + ncols - 1) // ncols
    # Increase figure width to accommodate legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols+2, 5*nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    for i, param in enumerate(params):
        ax = axes[i]
        for method in METHODS:
            plot_parameter(ax, data, method, set_name, param, dims)
        ax.set_title(PARAM_TITLES.get(param, param), fontsize=12)
        ax.set_xlabel('Dimension $d$', fontsize=10)
        ax.set_ylabel('Rejection Rate (5% level)', fontsize=10)
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # Use 2 rows for legend - 7 columns in first row, 6 in second
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=9, fontsize=9, framealpha=1.0, edgecolor='black')

    # Adjust layout to make room for the 2-row legend
    # fig.suptitle(SET_TITLES[set_name], fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at top for 2-row legend
    return fig


def save_set_pdfs(data, save_dir):
    """Generate the requested PDFs: Set_0_5percent.pdf ... Set_5_5percent.pdf."""
    os.makedirs(save_dir, exist_ok=True)

    for set_name, params in SET_PARAMS.items():
        ncols = 2 if set_name == 'Set 0' else 3
        fig = create_figure_with_grid(set_name, params, data, DIMENSIONS, ncols)
        pdf_path = os.path.join(save_dir, f"{set_name.replace(' ', '_')}_5percent.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"Saved: {pdf_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    if not os.path.isfile(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found.")
        return

    print("Parsing data...")
    data = parse_results(DATA_FILE)

    print("Generating PDFs (Set_0_5percent.pdf ... Set_5_5percent.pdf)...")
    save_set_pdfs(data, SAVE_DIR)

    print("All plots completed.")


if __name__ == "__main__":
    main()