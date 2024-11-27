# plot_test_results.py

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_test_results(results_folder, num_samples=50000):
    # Paths to the JSON files
    y_true_path = os.path.join(results_folder, "y_true_descaled.json")
    y_pred_path = os.path.join(results_folder, "y_pred_descaled.json")
    metrics_path = os.path.join(results_folder, "results.json")

    # Check if files exist
    if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
        print("Prediction files not found in the specified results folder.")
        return

    # Load data from JSON files
    with open(y_true_path, "r") as f:
        y_true_descaled = np.array(json.load(f))

    with open(y_pred_path, "r") as f:
        y_pred_descaled = np.array(json.load(f))

    # Load metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: R2 = {value['R2']:.4f}, MSE = {value['Test_MSE']:.6f}")
    else:
        print("Metrics file not found.")

    output_names = ["OSNR", "SNR", "BER"]

    print("Plotting the results...")

    for i in range(y_true_descaled.shape[1]):
        print(f"Processing {output_names[i]}...")

        y_true = y_true_descaled[:, i]
        y_pred = y_pred_descaled[:, i]

        # Filter out invalid values
        valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred) & \
                        ~np.isinf(y_true) & ~np.isinf(y_pred)

        y_true_valid = y_true[valid_indices]
        y_pred_valid = y_pred[valid_indices]

        num_valid_points = y_true_valid.shape[0]
        print(f"Number of valid points: {num_valid_points}")

        # Sample the data if there are too many points
        if num_valid_points > num_samples:
            sampled_indices = np.random.choice(num_valid_points, size=num_samples, replace=False)
            y_true_plot = y_true_valid[sampled_indices]
            y_pred_plot = y_pred_valid[sampled_indices]
        else:
            y_true_plot = y_true_valid
            y_pred_plot = y_pred_valid

        if len(y_true_plot) == 0 or len(y_pred_plot) == 0:
            print(f"No valid data to plot for {output_names[i]}. Skipping plot.")
            continue

        y_true_plot = np.array(y_true_plot, dtype=np.float64)
        y_pred_plot = np.array(y_pred_plot, dtype=np.float64)

        # Plot
        plt.figure()
        plt.scatter(y_true_plot, y_pred_plot, alpha=0.5)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{output_names[i]} Prediction")
        plt.grid(True)
        plt.tight_layout()
        filename = f"{output_names[i]}_results.png"
        plt.savefig(os.path.join(results_folder, filename))
        plt.close()
        print(f"Plot saved: {filename}")

if __name__ == "__main__":
    results_root = "lightpath_training/results"
    results_dirs = [d for d in os.listdir(results_root) if d.startswith("results_")]
    if not results_dirs:
        print("No results directories found.")
    else:
        results_dirs.sort()
        latest_results_folder = os.path.join(results_root, results_dirs[-1])
        print(f"Using results from: {latest_results_folder}")
        plot_test_results(latest_results_folder)
