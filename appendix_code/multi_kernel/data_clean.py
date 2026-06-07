import re

def parse_and_write_txt(input_file, output_file):
    """
    Parse results_for_plotting.txt, extract the 5% significance level rejection rates,
    and write them to a formatted text file.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    i = 0
    current_block = None
    current_set = None
    current_condition = None

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Detect block header: lines starting with '####################################################################'
        if line.startswith('####################################################################'):
            # Next non-empty line is the block description (e.g., "# single,d=1, balanced")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                block_desc = lines[i].strip()
                # Remove the leading '# ' if present
                if block_desc.startswith('#'):
                    block_desc = block_desc[1:].strip()
                current_block = block_desc
                output_lines.append(f"\n{'='*70}")
                output_lines.append(f"BLOCK: {current_block}")
                output_lines.append(f"{'='*70}")
                i += 1
            continue

        # Detect Set line (e.g., "Set 0, Null Distribution")
        if line.startswith('Set'):
            current_set = line
            output_lines.append(f"\n{current_set}")
            output_lines.append('-' * 50)
            i += 1
            continue

        # Detect condition line (e.g., "dgp=1:", "\mu=0.05, \sigma^2=0.5:", "df=3:", etc.)
        # Condition lines end with ':' and are not 'd=...' lines
        if line.endswith(':') and not line.startswith('d='):
            current_condition = line.rstrip(':')
            output_lines.append(f"\n{current_condition}:")
            i += 1
            continue

        # Detect lines containing dimension and tuple: "d=50: (0.092, 0.041, 0.012)"
        if line.startswith('d='):
            # Pattern: d=50: (0.092, 0.041, 0.012)
            match = re.match(r'd=(\d+):\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
            if match:
                d_val = int(match.group(1))
                # Middle value is the 5% rejection rate
                rate_5pct = float(match.group(3).strip())
                output_lines.append(f"  d={d_val:4d}: {rate_5pct:.3f}")
        i += 1

    # Write all collected lines to the output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Extracted 5% rejection rates written to {output_file}")

if __name__ == "__main__":
    input_path = "results_for_plotting.txt"
    output_path = "extracted_5pct_rates.txt"
    parse_and_write_txt(input_path, output_path)