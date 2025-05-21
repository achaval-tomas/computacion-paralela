#!/bin/python3
import subprocess, os
import csv
import re

num_threads = ["1"]
# Compilers to test
compilers = ["./runclang"]
# Values of N
n_values = [2**n for n in range(6, 15)]
# Output file
output_file = "archp_results.csv"

# Regex pattern to extract values
pattern = re.compile(r"total_cells_per_us: (\S+)")

# Run benchmarks and collect data
data = []
for threads in num_threads:
	my_env = os.environ.copy()
	my_env["OMP_NUM_THREADS"] = threads
	for compiler in compilers:
		for N in n_values:
			try:
				result = subprocess.run([compiler, str(N)], capture_output=True, text=True, check=True, env=my_env)
				match = pattern.search(result.stdout)
				if match:
					total_cells = float(match.group(1))
					data.append([compiler, threads, N, total_cells])
			except subprocess.CalledProcessError as e:
				print(f"Error running {compiler} with N={N}: {e}")

# Write data to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Compiler", "Num threads", "N", "total_cells_per_us"])
    writer.writerows(data)

print(f"Benchmark results saved to {output_file}")
