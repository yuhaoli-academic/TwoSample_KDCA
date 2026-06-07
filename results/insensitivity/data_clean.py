#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

INPUT_FILE = "insensitive_results.txt"
OUTPUT_FILE = "insensitive_results_5percent.txt"

def process_line(line):
    """Replace a triplet line with the middle (5%) value."""
    stripped = line.strip()
    if not stripped:
        return line

    # Pattern for lines like: d=50: (0.107, 0.058, 0.01)
    triplet_match = re.match(r'(d=\d+:\s*)\(([\d.,\s]+)\)', stripped)
    if triplet_match:
        prefix = triplet_match.group(1)
        values_str = triplet_match.group(2)
        # Split by comma and convert to float
        values = [float(v.strip()) for v in values_str.split(',')]
        if len(values) == 3:
            middle_value = values[1]   # 5% level
            new_line = f"{prefix}({middle_value:.3f})"
            # Replace only the matched part, preserving original indentation
            return line.replace(triplet_match.group(0), new_line)
    return line

def main():
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    output_lines = [process_line(line) for line in lines]

    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(output_lines)

    print(f"Extracted 5% results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()