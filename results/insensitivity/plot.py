#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# Configuration
# ============================================================================
DATA_FILE = "insensitive_results_5percent.txt"
SAVE_DIR = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/draft/arxiv_v2/art/insensitivity/"

# All method variants (9 total)
METHODS = [
    "TMMD(d=1) (median)", "TMMD(d=1) (100)", "TMMD(d=1) (200)",
    "MMD (median)", "MMD (100)", "MMD (200)",
    "TMMD-Oracle (median)", "TMMD-Oracle (100)", "TMMD-Oracle (200)"
]

# Mapping from raw header to method key
HEADER_TO_METHOD = {
    "TMMD m = 100; n = 100; sigma = median": "TMMD(d=1) (median)",
    "TMMD m = 100; n = 100; sigma=100.0": "TMMD(d=1) (100)",
    "TMMD m = 100; n = 100; sigma=200.0": "TMMD(d=1) (200)",
    "MMD Permutationmutation m = 100; n = 100; sigma=median": "MMD (median)",
    "MMD Permutation m = 100; n = 100; sigma=100": "MMD (100)",
    "MMD Permutation m = 100; n = 100; sigma=200": "MMD (200)",
    "TMMD-Oracle; sigma = median": "TMMD-Oracle (median)",
    "TMMD-Oracle; sigma = 100": "TMMD-Oracle (100)",
    "TMMD-Oracle; sigma = 200": "TMMD-Oracle (200)"
}

# Category colors (one per category, variants use same color with different markers)
CATEGORY_COLORS = {
    "TMMD": "#E69F00",   # orange
    "MMD": "#d62728",    # red
    "TMMD-Oracle": "#2ca02c"  # green
}

# Marker styles for variants within each category
CATEGORY_MARKERS = {
    "TMMD": ["o", "s", "^"],
    "MMD": ["D", "v", "p"],
    "TMMD-Oracle": ["*", "h", "X"]
}

# Build style dict for each method
METHOD_STYLES = {}
for idx, method in enumerate(METHODS):
    # Fix: Check for Oracle first to avoid matching with "TMMD"
    if method.startswith("TMMD-Oracle"):
        cat = "TMMD-Oracle"
    elif method.startswith("TMMD"):
        cat = "TMMD"
    elif method.startswith("MMD"):
        cat = "MMD"
    else:
        cat = "TMMD-Oracle"  # fallback (should not happen)
    
    variant_idx = [m for m in METHODS if m.startswith(cat)].index(method)
    METHOD_STYLES[method] = {
        "color": CATEGORY_COLORS[cat],
        "marker": CATEGORY_MARKERS[cat][variant_idx],
        "linestyle": "-" if variant_idx == 0 else "--" if variant_idx == 1 else ":",
        "linewidth": 1.5,
        "markersize": 6
    }

# Display names (can be same as method keys)
METHOD_LABELS = {m: m for m in METHODS}

DIMENSIONS = [50, 100, 500, 1000]

# Parameter sets (same as in template)
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
# Parsing functions
# ============================================================================
def canonicalize_param(param_str):
    """Normalize parameter strings (remove spaces around equals, etc.)."""
    param_str = param_str.strip()
    param_str = re.sub(r'\\sigma\^2\s*=\s*', r'\\sigma^2=', param_str)
    param_str = re.sub(r'\s*,\s*', ', ', param_str)
    return param_str

def parse_results(filepath):
    """Parse the uniform 5% data file into nested dict."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {method: {} for method in METHODS}
    current_method = None
    current_set = None
    current_param = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Method header
        if line in HEADER_TO_METHOD:
            current_method = HEADER_TO_METHOD[line]
            data[current_method] = {}
            current_set = None
            current_param = None
            continue

        # Set header (e.g., "Set 0, Null Distribution")
        set_match = re.match(r'Set ([0-5])[,:]?', line)
        if set_match and current_method:
            current_set = f"Set {set_match.group(1)}"
            data[current_method][current_set] = {}
            current_param = None
            continue

        # Parameter line
        param_line = None
        if line.startswith('dgp='):
            param_line = line.split(':')[0].strip()
        elif line.startswith('df='):
            param_line = line.split(':')[0].strip()
        elif line.startswith(r'\mu=') or line.startswith('\\mu='):
            raw = line.split(':')[0].strip().replace('\\mu', r'\mu')
            param_line = canonicalize_param(raw)
        elif line.startswith(r'\sigma') or line.startswith('\\sigma'):
            raw = line.split(':')[0].strip().replace('\\sigma', r'\sigma')
            param_line = canonicalize_param(raw)

        if param_line and current_method and current_set:
            current_param = param_line
            data[current_method][current_set][current_param] = {}
            continue

        # Data line: d=50: (0.058)
        data_match = re.match(r'd=(\d+):\s*\(([\d.]+)\)', line)
        if data_match and current_method and current_set and current_param:
            dim = int(data_match.group(1))
            value = float(data_match.group(2))
            data[current_method][current_set][current_param][dim] = value

    return data


# ============================================================================
# Plotting functions
# ============================================================================
def plot_method(ax, data, method, set_name, param, dims):
    """Plot one method's rejection rates for given set and parameter."""
    if method not in data:
        return
    if set_name not in data[method]:
        return
    if param not in data[method][set_name]:
        return
    param_dict = data[method][set_name][param]
    rates = [param_dict.get(d, None) for d in dims]
    if any(r is not None for r in rates):
        style = METHOD_STYLES[method]
        ax.plot(dims, rates,
                marker=style['marker'],
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                label=METHOD_LABELS[method])

def create_figure_for_set(set_name, params, data, dims, ncols):
    """Create a figure with subplots arranged in grid."""
    n_params = len(params)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols + 5, 5*nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    for i, param in enumerate(params):
        ax = axes[i]
        for method in METHODS:
            plot_method(ax, data, method, set_name, param, dims)
        ax.set_title(PARAM_TITLES.get(param, param), fontsize=12)
        ax.set_xlabel('Dimension $d$', fontsize=10)
        ax.set_ylabel('Rejection Rate (5% level)', fontsize=10)
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Legend (only once per figure)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.005),
                   ncol=8, fontsize=9, edgecolor='black')

    fig.suptitle(SET_TITLES[set_name], fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

def save_pdfs(data, save_dir):
    """Generate PDFs for all sets."""
    os.makedirs(save_dir, exist_ok=True)
    for set_name, params in SET_PARAMS.items():
        ncols = 2 if set_name == 'Set 0' else 3
        fig = create_figure_for_set(set_name, params, data, DIMENSIONS, ncols)
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

    print("Generating PDFs...")
    save_pdfs(data, SAVE_DIR)

    print("All plots completed.")

if __name__ == "__main__":
    main()