import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------------------------------------------------
# 0. Style configuration
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 1. Parser (handles "BLOCK: Set 0, ..." correctly)
# ----------------------------------------------------------------------
def parse_rates_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    method_headers = [
        "BLOCK: single,d=1, balanced",
        "BLOCK: multiple,d=1, balanced",
        "BLOCK: single, learn, balanced",
        "BLOCK: multiple, learn, balanced"
    ]
    method_names = {
        "BLOCK: single,d=1, balanced": "TMMD(d=1)",
        "BLOCK: multiple,d=1, balanced": "MultiTMMD(d=1)",
        "BLOCK: single, learn, balanced": "TMMD-Oracle",
        "BLOCK: multiple, learn, balanced": "MultiTMMD-Oracle"
    }

    # Split content into method blocks
    blocks = []
    pos = 0
    while pos < len(content):
        next_header = None
        next_pos = len(content)
        for hdr in method_headers:
            p = content.find(hdr, pos)
            if p != -1 and p < next_pos:
                next_header = hdr
                next_pos = p
        if next_header is None:
            break
        block_end = len(content)
        for hdr in method_headers:
            p = content.find(hdr, next_pos + len(next_header))
            if p != -1 and p < block_end:
                block_end = p
        block_text = content[next_pos:block_end]
        blocks.append((next_header, block_text))
        pos = block_end

    data = []
    set_header_pattern = re.compile(r'^(?:BLOCK:\s*)?Set\s+(\d+)(?:,\s*(.+))?$', re.MULTILINE)
    dgp_pattern = re.compile(r'^dgp=(\d+):$', re.MULTILINE)
    value_pattern = re.compile(r'^\s*d=\s*(\d+):\s*([\d\.]+)$', re.MULTILINE)

    for header, block in blocks:
        method = method_names[header]
        lines = block.splitlines()
        current_set = None
        current_scenario = None
        set_id = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('===') or line.startswith('---'):
                continue

            set_match = set_header_pattern.match(line)
            if set_match:
                set_id = int(set_match.group(1))
                desc = set_match.group(2).strip() if set_match.group(2) else ""
                current_set = f"Set {set_id}"
                current_scenario = None
                continue

            dgp_match = dgp_pattern.match(line)
            if dgp_match and current_set and set_id == 0:
                current_scenario = f"dgp={dgp_match.group(1)}"
                continue

            # For other sets: scenario description lines (keep exactly as in file)
            if line.endswith(':') and 'd=' not in line and current_set and set_id != 0:
                scenario_raw = line.rstrip(':')
                current_scenario = scenario_raw.strip()
                continue

            val_match = value_pattern.match(line)
            if val_match and method and current_set and current_scenario is not None:
                d_val = int(val_match.group(1))
                rate = float(val_match.group(2))
                data.append({
                    'method': method,
                    'set': current_set,
                    'scenario': current_scenario,
                    'd': d_val,
                    'rate': rate
                })

    return pd.DataFrame(data)


# ----------------------------------------------------------------------
# 2. Helper: plot one subplot (no legend inside)
# ----------------------------------------------------------------------
def plot_scenario_axes(ax, df_scenario, scenario_name, set_name, ylim_set0=False):
    # Use the same order as before
    methods_order = ["TMMD(d=1)", "MultiTMMD(d=1)", "TMMD-Oracle", "MultiTMMD-Oracle"]
    colors = ['blue', 'orange', 'green', 'red']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    for idx, method in enumerate(methods_order):
        method_df = df_scenario[df_scenario['method'] == method].sort_values('d')
        if method_df.empty:
            continue
        ax.plot(method_df['d'], method_df['rate'],
                color=colors[idx], marker=markers[idx], linestyle=linestyles[idx],
                linewidth=1.5, markersize=5, label=method)

    ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks(DATA_DIMS)
    ax.set_xticklabels(DATA_DIMS)
    ax.set_xlabel('Data dimension $d$', fontsize=9)
    ax.set_ylabel('Rejection Rate (5% level)', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.3)
    if ylim_set0:
        ax.set_ylim(0, 0.1)


# ----------------------------------------------------------------------
# 3. Produce PDFs – using reference style for subplot sizing
# ----------------------------------------------------------------------
def produce_pdfs(df):
    for set_key, scenarios_ordered in SET_PARAMS.items():
        set_df = df[df['set'] == set_key]
        if set_df.empty:
            print(f"Warning: No data found for {set_key}")
            continue

        n_scenarios = len(scenarios_ordered)
        # Determine grid layout
        if set_key == 'Set 0':
            ncols = 2
            nrows = 2
            ylim_set0 = True
        else:
            ncols = 3
            nrows = (n_scenarios + ncols - 1) // ncols
            ylim_set0 = False

        # Figure size: width = 5*ncols + 2, height = 4*nrows
        fig_width = 5 * ncols + 2
        fig_height = 4 * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot each scenario
        for idx, scen_raw in enumerate(scenarios_ordered):
            scen_df = set_df[set_df['scenario'] == scen_raw]
            if scen_df.empty:
                print(f"Warning: No data for scenario '{scen_raw}' in {set_key}")
                axes[idx].set_visible(False)
                continue
            pretty_title = PARAM_TITLES.get(scen_raw, scen_raw)
            plot_scenario_axes(axes[idx], scen_df, pretty_title, set_key, ylim_set0=ylim_set0)
            axes[idx].set_title(pretty_title, fontsize=10, pad=6)

        # Hide any unused subplots
        for j in range(len(scenarios_ordered), len(axes)):
            axes[j].set_visible(False)

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
                       ncol=4, fontsize=9, frameon=False)

        fig.suptitle(SET_TITLES[set_key], fontsize=14, y=0.98)
        plt.tight_layout()
        # Save with bbox_inches='tight' to include the legend
        safe_name = set_key.replace(' ', '_')
        pdf_filename = f"{safe_name}.pdf"
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {pdf_filename}")


# ----------------------------------------------------------------------
# 4. Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    input_file = "extracted_5pct_rates.txt"
    df_rates = parse_rates_file(input_file)
    print(f"Parsed {len(df_rates)} rows.")
    if len(df_rates) > 0:
        print(df_rates.head())
        produce_pdfs(df_rates)
    else:
        print("No data parsed. Check file path.")