#!/bin/python3
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run_program(program, K, num_runs):
    """Run the program num_runs times and return the average 'cells_per_us' value."""
    cells_per_us_values = []
    
    for _ in range(num_runs):
        # Run the program and capture the output
        result = subprocess.run([program, str(K), str(0.1), str(0.0), str(0.0), str(5.0), str(100.0)], capture_output=True, text=True)
        
        # Debug: Print the raw output of the program for troubleshooting
        # print(f"Running {program} with K={K}...")
        # print("Program output:\n", result.stdout)
        
        # Extract the value of 'cells_per_us' from the output
        output = result.stdout
        found_value = False  # Flag to check if we found the expected output
        for line in output.splitlines():
            if "cells_per_us:" in line:
                try:
                    cells_per_us = float(line.split(":")[1].strip())
                    cells_per_us_values.append(cells_per_us)
                    found_value = True
                except ValueError:
                    print(f"Error parsing 'cells_per_us' value from line: {line}")
        
        # If no valid cells_per_us was found, print an error
        if not found_value:
            print(f"Warning: No 'cells_per_us' value found for K={K}")
    
    # Return the average value if there are results
    if cells_per_us_values:
        return np.mean(cells_per_us_values)
    else:
        return None  # Return None if no valid data was found

def plot_results(K_values, headless_averages, intel_headless_averages):
    """Plot the results of both programs."""
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, headless_averages, label="./headless", color='blue', marker='o')
    plt.plot(K_values, intel_headless_averages, label="./intel-headless", color='red', marker='x')
    
    plt.xlabel("K (Value Passed to Programs)")
    plt.ylabel("Average cells_per_us")
    plt.title("Comparison of Average cells_per_us for ./headless and ./intel-headless")
    plt.legend()
    plt.xscale("log")  # Log scale for K-axis (since K values vary over orders of magnitude)
    plt.grid(True)
    plt.savefig("plot-res")

def main():
    K_values = [10, 100, 200, 300]  # K values to test
    num_runs = [100, 5, 3, 1]
    headless_averages = []  # To store average values for ./headless
    intel_headless_averages = []  # To store average values for ./intel-headless

    for K, num_runs in zip(K_values, num_runs):
        # Run and average the results for ./headless
        print(f"Running './headless' with K={K}, num_runs={num_runs}...")
        headless_avg = run_program("./headless", K, num_runs)
        headless_averages.append(headless_avg if headless_avg is not None else np.nan)  # Append average or NaN if failed
        
        # Run and average the results for ./intel-headless
        print(f"Running './intel-headless' with K={K}, num_runs={num_runs}...")
        intel_headless_avg = run_program("./intel-headless", K, num_runs)
        intel_headless_averages.append(intel_headless_avg if intel_headless_avg is not None else np.nan)  # Append average or NaN if failed
    
    # Plot the results
    plot_results(K_values, headless_averages, intel_headless_averages)

if __name__ == "__main__":
    main()
