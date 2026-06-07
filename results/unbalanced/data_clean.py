import re

def parse_and_write_txt(input_file, output_file):
    """
    Parse the results file, extract the 5% significance level rejection rates,
    and write them to a formatted text file.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    current_method = None
    current_set = None
    current_condition = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Detect method name (line starting with "====" followed by method name)
        if line.startswith("====") and i+1 < len(lines) and not lines[i+1].strip().startswith("===="):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                method_candidate = lines[j].strip()
                if not method_candidate.startswith("===="):
                    current_method = method_candidate
                    output_lines.append(f"\n{'='*60}")
                    output_lines.append(f"{current_method}")
                    output_lines.append(f"{'='*60}")
                    i = j + 1
                    continue
        
        # Detect a "Set" header line
        if line.startswith("Set"):
            current_set = line
            output_lines.append(f"\n{current_set}")
            output_lines.append("-" * 40)
            i += 1
            continue
        
        # Detect a condition line (e.g., "dgp=1:", or "\mu=0.05, \sigma^2=0.5:")
        if '=' in line and not line.startswith('d=') and not line.startswith('Set'):
            current_condition = line.rstrip(':')
            output_lines.append(f"{current_condition}:")
            i += 1
            continue
        
        # Detect a line with dimension and tuple of rejection rates
        if line.startswith('d='):
            m = re.match(r'd=(\d+):\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
            if m:
                d = int(m.group(1))
                # The middle value is the 5% significance level
                rate_5pct = float(m.group(3).strip())
                output_lines.append(f"  d={d:4d}: {rate_5pct:.3f}")
        i += 1
    
    # Write all collected lines to the output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Extracted data written to {output_file}")


if __name__ == "__main__":
    input_path = "results_for_plotting.txt"
    output_path = "extracted_5pct_rates.txt"
    parse_and_write_txt(input_path, output_path)