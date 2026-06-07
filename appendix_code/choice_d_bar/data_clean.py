#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv

# ============================================================================
# Configuration
# ============================================================================
INPUT_FILE = "d_bar_results.txt"
OUTPUT_CSV = "extracted_5percent_rates.csv"

# All data dimensions present in the file
DATA_DIMS = [50, 100, 500, 1000]

# ============================================================================
# Robust parser
# ============================================================================
def parse_dbar_file(filepath):
    """
    Returns a list of dictionaries, each containing:
        d_bar, set_name, param, data_dim, rate_5percent
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    records = []
    current_dbar = None
    current_set = None
    current_param = None

    # Regular expressions
    dbar_pattern = re.compile(r'\\bar\{d\}=(\d+)')
    set_pattern = re.compile(r'^Set (\d+),')
    param_pattern = re.compile(r'^(dgp=\d+|df=\d+|\\mu=.*|\\sigma\^2=.*):$')
    data_pattern = re.compile(r'd=(\d+):\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect \bar{d}=3 (appears between === lines)
        if '\\bar{d}=' in line:
            m = dbar_pattern.search(line)
            if m:
                current_dbar = int(m.group(1))
                current_set = None
                current_param = None
            continue

        # Detect Set header, e.g. "Set 0, Null Distribution"
        if line.startswith('Set ') and ',' in line:
            m = set_pattern.match(line)
            if m:
                current_set = f"Set {m.group(1)}"
                current_param = None
            continue

        # Detect parameter line, e.g. "dgp=1:" or "\mu=0.05, \sigma^2=0.5:"
        if param_pattern.match(line):
            # Remove trailing colon and strip
            current_param = line.rstrip(':').strip()
            # Normalize LaTeX (remove spaces around = and commas)
            current_param = re.sub(r'\\sigma\^2\s*=\s*', r'\\sigma^2=', current_param)
            current_param = re.sub(r'\\mu\s*=\s*', r'\\mu=', current_param)
            current_param = re.sub(r'\s*,\s*', ', ', current_param)
            continue

        # Detect data line: d=50: (0.112, 0.052, 0.01)
        m = data_pattern.match(line)
        if m and current_dbar is not None and current_set is not None and current_param is not None:
            data_dim = int(m.group(1))
            # m.group(3) is the second number (5% rejection rate)
            rate_5percent = float(m.group(3))
            records.append({
                'd_bar': current_dbar,
                'Set': current_set,
                'Parameter': current_param,
                'Data dimension': data_dim,
                '5% rejection rate': rate_5percent
            })

    return records

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Parsing {INPUT_FILE} ...")
    records = parse_dbar_file(INPUT_FILE)

    if not records:
        print("Warning: No data extracted. Check file format and patterns.")
        return

    print(f"Extracted {len(records)} records.")

    # Write to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['d_bar', 'Set', 'Parameter', 'Data dimension', '5% rejection rate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()