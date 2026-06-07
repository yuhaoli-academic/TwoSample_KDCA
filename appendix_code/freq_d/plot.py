#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# Configuration
# ============================================================================
DATA_FILE = "frequency_of_d.txt"
SAVE_DIR = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/draft/arxiv_v2/art/freq_d/"

# Data dimensions (sample sizes)
DATA_DIMS = [50, 100, 500, 1000]
SELECTED_DIMS = [1, 2, 3, 4, 5]   # possible selected d

# Define sets and their parameters (order as in file)
SET_PARAMS = {
    'Set 0': ['dgp=1', 'dgp=2', 'dgp=3', 'dgp=4'],
    'Set 1': [r'\mu=0.05, \sigma^2=0.5', r'\mu=0.1, \sigma^2=1.3', r'\mu=-0.05, \sigma^2=0.6'],
    'Set 2': ['df=3', 'df=5', 'df=10'],
    'Set 3': [r'\mu=-0.05, \sigma^2=0.85', r'\mu=0.0, \sigma^2=1.1', r'\mu=0.05, \sigma^2=1.05'],
    'Set 4': [r'\sigma^2=0.6', r'\sigma^2=0.8', r'\sigma^2=1.3'],
    'Set 5': [r'\mu=-1.0', r'\mu=0.6', r'\mu=1.3']
}

# Nicer titles for subplots (optional)
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

# Colors for the four data dimensions
DIM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
DIM_LABELS = [f'dim = {d}' for d in DATA_DIMS]

# ============================================================================
# Parsing functions
# ============================================================================
def parse_frequencies(filepath):
    """
    Parse the text file into a nested dictionary:
    data[set_name][param][data_dim][selected_d] = frequency
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}
    current_set = None
    current_param = None

    # Regex to match lines like: d=50: d_freq = {1: np.float64(0.327), ...}
    pattern = re.compile(
        r'd=(\d+):\s*d_freq\s*=\s*\{([^}]+)\}'
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Set header: "Set 0, Null Distribution"
        if line.startswith('Set ') and ',' in line:
            # Extract set name (e.g., "Set 0")
            current_set = line.split(',')[0].strip()
            data[current_set] = {}
            current_param = None
            continue

        # Parameter line: e.g., "dgp=1:" or "\mu=0.05, \sigma^2=0.5:"
        if current_set is not None and ':' in line and not line.startswith('d='):
            # Check if line ends with colon and contains a parameter name
            # It might start with dgp=, df=, \mu=, \sigma^2=
            if line.startswith('dgp=') or line.startswith('df=') or \
               line.startswith(r'\mu=') or line.startswith(r'\sigma^2='):
                param = line.rstrip(':').strip()
                # Normalize LaTeX (remove extra spaces)
                param = re.sub(r'\\sigma\^2\s*=\s*', r'\\sigma^2=', param)
                param = re.sub(r'\\mu\s*=\s*', r'\\mu=', param)
                param = re.sub(r'\s*,\s*', ', ', param)
                data[current_set][param] = {}
                current_param = param
                continue

        # Data line: d=50: d_freq = {...}
        match = pattern.search(line)
        if match and current_set and current_param:
            data_dim = int(match.group(1))
            freq_dict_str = match.group(2)
            # Parse dictionary: "1: np.float64(0.327), 2: np.float64(0.242), ..."
            freq_pairs = re.findall(r'(\d+):\s*np\.float64\(([\d.]+)\)', freq_dict_str)
            freq = {int(k): float(v) for k, v in freq_pairs}
            # Store in nested dict
            if data_dim not in data[current_set][current_param]:
                data[current_set][current_param][data_dim] = {}
            data[current_set][current_param][data_dim] = freq

    return data

# ============================================================================
# Plotting functions
# ============================================================================
def create_figure_for_set(set_name, params, data, ncols):
    """
    Create a figure with subplots for each parameter in the set.
    For Set 0: 2x2 grid; others: 1x3 grid.
    """
    n_params = len(params)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols + 2, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    bar_width = 0.2
    x = np.arange(len(SELECTED_DIMS))  # positions for selected dims (1..5)

    for i, param in enumerate(params):
        ax = axes[i]
        param_data = data.get(set_name, {}).get(param, {})
        if not param_data:
            ax.text(0.5, 0.5, f'No data for {param}', ha='center', va='center')
            ax.set_title(PARAM_TITLES.get(param, param))
            continue

        # For each data dimension, extract frequencies for selected dims 1..5
        for j, data_dim in enumerate(DATA_DIMS):
            freq_dict = param_data.get(data_dim, {})
            freqs = [freq_dict.get(d, 0.0) for d in SELECTED_DIMS]
            offset = (j - len(DATA_DIMS)/2 + 0.5) * bar_width
            ax.bar(x + offset, freqs, bar_width,
                   label=DIM_LABELS[j], color=DIM_COLORS[j], edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(SELECTED_DIMS)
        ax.set_xlabel('Selected dimension $d$')
        ax.set_ylabel('Frequency')
        ax.set_ylim(0, 1.05)
        ax.set_title(PARAM_TITLES.get(param, param))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a common legend (only once per figure)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.00),
                   ncol=len(DATA_DIMS), fontsize=9)

    fig.suptitle(SET_TITLES[set_name], fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

def save_pdfs(data, save_dir):
    """Generate PDFs for all sets."""
    os.makedirs(save_dir, exist_ok=True)
    for set_name, params in SET_PARAMS.items():
        ncols = 2 if set_name == 'Set 0' else 3
        fig = create_figure_for_set(set_name, params, data, ncols)
        pdf_path = os.path.join(save_dir, f"{set_name.replace(' ', '_')}_freq_d.pdf")
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
    data = parse_frequencies(DATA_FILE)

    print("Generating PDFs...")
    save_pdfs(data, SAVE_DIR)

    print("All plots completed.")

if __name__ == "__main__":
    main()