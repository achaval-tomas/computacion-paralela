#!/bin/python3
import subprocess
import csv
import re

# Compilers to test
compilers = ["./rungcc", "./runclang", "./runintel"]
# Values of N
n_values = [128, 256, 512, 1024, 2048, 4096]
# Output file
output_file = "benchmark_results.csv"

# Regex pattern to extract values
pattern = re.compile(r"react_cells_per_us: (\S+)\nvel_cells_per_us: (\S+)\ndens_cells_per_us: (\S+)\ntotal_cells_per_us: (\S+)")

# Run benchmarks and collect data
data = []
for compiler in compilers:
    for N in n_values:
        try:
            result = subprocess.run([compiler, str(N), str(0.1), str(0.0), str(0.0), str(5.0), str(100.0)], capture_output=True, text=True, check=True)
            match = pattern.search(result.stdout)
            if match:
                r_cells, v_cells, d_cells, total_cells = map(float, match.groups())
                data.append([compiler, N, r_cells, v_cells, d_cells, total_cells])
        except subprocess.CalledProcessError as e:
            print(f"Error running {compiler} with N={N}: {e}")

# Write data to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Compiler", "N", "react_cells_per_us", "vel_cells_per_us", "dens_cells_per_us", "total_cells_per_us"])
    writer.writerows(data)

print(f"Benchmark results saved to {output_file}")
