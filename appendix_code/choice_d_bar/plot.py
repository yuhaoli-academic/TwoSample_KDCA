#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# Configuration
# ============================================================================
CSV_FILE = "extracted_5percent_rates.csv"
OUTPUT_DIR = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/draft/arxiv_v2/art/d_bar_choice/"

DATA_DIMS = [50, 100, 500, 1000]

SET_PARAMS = {
    'Set 0': ['dgp=1', 'dgp=2', 'dgp=3', 'dgp=4'],
    'Set 1': [r'\mu=0.05, \sigma^2=0.5', r'\mu=0.1, \sigma^2=1.3', r'\mu=-0.05, \sigma^2=0.6'],
    'Set 2': ['df=3', 'df=5', 'df=10'],
    'Set 3': [r'\mu=-0.05, \sigma^2=0.85', r'\mu=0.0, \sigma^2=1.1', r'\mu=0.05, \sigma^2=1.05'],
    'Set 4': [r'\sigma^2=0.6', r'\sigma^2=0.8', r'\sigma^2=1.3'],
    'Set 5': [r'\mu=-1.0', r'\mu=0.6', r'\mu=1.3']
}

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

DBAR_STYLES = {
    3: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': r'TMMD-Oracle($\bar{d}=3$)'},
    5: {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': r'TMMD-Oracle($\bar{d}=5$)'},
    7: {'color': '#2ca02c', 'linestyle': ':', 'marker': '^', 'label': r'TMMD-Oracle($\bar{d}=7$)'}
}

# ============================================================================
# Plotting function
# ============================================================================
def plot_comparison_figure(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for set_name, params in SET_PARAMS.items():
        n_params = len(params)
        ncols = 2 if set_name == 'Set 0' else 3
        nrows = (n_params + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols + 2, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        # Plot data on each subplot
        for i, param in enumerate(params):
            ax = axes[i]
            plotted = False
            for dbar, style in DBAR_STYLES.items():
                sub = df[(df['Set'] == set_name) & (df['Parameter'] == param) & (df['d_bar'] == dbar)]
                if sub.empty:
                    continue
                dims = []
                rates = []
                for dim in DATA_DIMS:
                    val = sub[sub['Data dimension'] == dim]['5% rejection rate'].values
                    if len(val) > 0:
                        dims.append(dim)
                        rates.append(val[0])
                if dims:
                    ax.plot(dims, rates,
                            color=style['color'],
                            linestyle=style['linestyle'],
                            marker=style['marker'],
                            linewidth=1.5,
                            markersize=6,
                            label=style['label'])
                    plotted = True
            if not plotted:
                ax.text(0.5, 0.5, f'No data for {param}', ha='center', va='center')
            ax.set_xlabel('Data dimension $d$')
            ax.set_ylabel('Rejection Rate (5% level)')
            ax.set_xscale('log')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            # ax.set_title(PARAM_TITLES.get(param, param), fontsize=10)

        # Collect unique legend handles from all subplots
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

        if handles:
            # Adjust figure to make room for legend at bottom
            fig.subplots_adjust(bottom=0.2)
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                       ncol=3, fontsize=9, frameon=False)

        fig.suptitle(SET_TITLES[set_name], fontsize=14, y=0.98)
        plt.tight_layout()
        # Save with bbox_inches='tight' to include the legend
        pdf_path = os.path.join(output_dir, f"{set_name.replace(' ', '_')}_comparison.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {pdf_path}")

# ============================================================================
# Main
# ============================================================================
def main():
    if not os.path.isfile(CSV_FILE):
        print(f"Error: CSV file '{CSV_FILE}' not found. Run extract_5percent.py first.")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} records from {CSV_FILE}")
    plot_comparison_figure(df, OUTPUT_DIR)
    print("All plots completed.")

if __name__ == "__main__":
    main()