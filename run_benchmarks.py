#!/bin/python3
import subprocess
import csv
import re

# Compilers to test
compilers = ["./rungcc", "./runclang", "./runintel"]
# Values of N
n_values = [64, 128, 256, 512, 1024]
# Output file
output_file = "archp_results.csv"

# Regex pattern to extract values
pattern = re.compile(r"total_cells_per_us: (\S+)")

# Run benchmarks and collect data
data = []
for compiler in compilers:
    for N in n_values:
        try:
            result = subprocess.run([compiler, str(N), str(0.1), str(0.0), str(0.0), str(5.0), str(100.0)], capture_output=True, text=True, check=True)
            match = pattern.search(result.stdout)
            if match:
                total_cells = float(match.group(1))
                data.append([compiler, N, total_cells])
        except subprocess.CalledProcessError as e:
            print(f"Error running {compiler} with N={N}: {e}")

# Write data to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Compiler", "N", "total_cells_per_us"])
    writer.writerows(data)

print(f"Benchmark results saved to {output_file}")
