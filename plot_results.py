#!/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

# Load data
input_file = "benchmark_results.csv"
df = pd.read_csv(input_file)

# Plot results
plt.figure(figsize=(10, 6))
for metric in ["react_cells_per_us", "vel_cells_per_us", "dens_cells_per_us", "total_cells_per_us"]:
    for compiler in df["Compiler"].unique():
        subset = df[df["Compiler"] == compiler]
        plt.plot(subset["N"], subset[metric], marker='o', linestyle='-', label=f"{compiler} - {metric}")

# Labels and title
plt.xlabel("N")
plt.ylabel("Performance (cells per µs)")
plt.title("Benchmark Results")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
