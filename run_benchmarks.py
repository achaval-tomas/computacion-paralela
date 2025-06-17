#!/bin/python3
import subprocess
import csv
import re

# Compilers to test
compilers = ["./runcuda"]
# Values of N and T
n_values = [2**i for i in range(6, 15)]
t_values = [2**i for i in range(5, 11)]
# Output file
output_file = "atom_cuda_results.csv"

# Regex pattern to extract values
pattern = re.compile(r"total_cells_per_us: (\S+)")

# Run benchmarks and collect data
data = []
for compiler in compilers:
    for N in n_values:
        for T in t_values:
            try:
                result = subprocess.run([compiler, str(N), str(T)], capture_output=True, text=True, check=True)
                match = pattern.search(result.stdout)
                if match:
                    total_cells = float(match.group(1))
                    data.append([compiler, N, T, total_cells])
            except subprocess.CalledProcessError as e:
                print(f"Error running {compiler} with N={N}: {e}")

# Write data to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Compiler", "N", "Threads_per_block", "total_cells_per_us"])
    writer.writerows(data)

print(f"Benchmark results saved to {output_file}")
