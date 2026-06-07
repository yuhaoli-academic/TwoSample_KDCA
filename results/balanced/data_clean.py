#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

INPUT_FILE = "results_for_plotting.txt"
OUTPUT_FILE = "results_for_plotting_uniform.txt"

def process_line(line):
    """
    Convert any data line to the uniform format: d=X: (value)
    Returns the modified line, or the original line if no conversion is needed.
    """
    stripped = line.strip()
    if not stripped:
        return line

    # Pattern 1: d=50: (0.065, 0.037, 0.005)   (triplet)
    # Handles variable whitespace inside parentheses.
    triplet_match = re.match(r'(d=\d+:\s*)\(([\d.,\s]+)\)', stripped)
    if triplet_match:
        prefix = triplet_match.group(1)
        values_str = triplet_match.group(2)
        values = [float(v.strip()) for v in values_str.split(',')]
        if len(values) == 3:
            # Keep only the 5% value (index 1)
            new_value = values[1]
            new_line = f"{prefix}({new_value:.3f})"
            return line.replace(triplet_match.group(0), new_line)
        else:
            return line

    # Pattern 2: d=50: Rej Rate at α=0.05: 0.064   (SpectraRegu format)
    # The 'α' character may appear as 'Î±' if the file was mis-decoded.
    # We match either 'α' or 'Î±' to be robust.
    spec_match = re.match(
        r'(d=\d+:\s*)Rej Rate at (?:α|Î±)=0\.05:\s*([\d.]+)',
        stripped
    )
    if spec_match:
        prefix = spec_match.group(1)
        value = float(spec_match.group(2))
        new_line = f"{prefix}({value:.3f})"
        return line.replace(spec_match.group(0), new_line)

    # No conversion needed
    return line

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_lines = []
    for line in lines:
        new_line = process_line(line)
        output_lines.append(new_line)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"Uniform 5% data file saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()