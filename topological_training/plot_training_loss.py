import os
import json
import matplotlib.pyplot as plt

def plot_latest_training_loss():
    # Define the root directory where loss directories are stored
    root_dir = "topological_training"

    # Find all directories starting with 'loss_training_'
    loss_dirs = [
        d for d in os.listdir(root_dir)
        if d.startswith('loss_training_') and os.path.isdir(os.path.join(root_dir, d))
    ]

    if not loss_dirs:
        print("No loss_training directories found.")
        return

    # Extract indices and find the latest directory
    loss_indices = [int(d.split('_')[-1]) for d in loss_dirs]
    latest_index = max(loss_indices)
    latest_loss_dir = f"loss_training_{latest_index}"
    loss_dir_path = os.path.join(root_dir, latest_loss_dir)

    print(f"Plotting from the latest loss directory: {latest_loss_dir}")

    # Read the loss and metrics from JSON files
    with open(os.path.join(loss_dir_path, "loss_history.json"), "r") as f:
        loss_history = json.load(f)

    with open(os.path.join(loss_dir_path, "val_loss_history.json"), "r") as f:
        val_loss_history = json.load(f)

    with open(os.path.join(loss_dir_path, "r2_history.json"), "r") as f:
        r2_history = json.load(f)

    with open(os.path.join(loss_dir_path, "val_r2_history.json"), "r") as f:
        val_r2_history = json.load(f)

    # Ensure the lists are of the same length
    epochs = range(1, len(loss_history) + 1)

    # Plot Loss Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_history, 'b-', label="Training Loss")
    plt.plot(epochs, val_loss_history, 'r-', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the loss plot in the same directory
    loss_plot_path = os.path.join(loss_dir_path, f"loss_plot_{latest_index}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

    # Plot R2 Score Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, r2_history, 'b-', label="Training R2 Score")
    plt.plot(epochs, val_r2_history, 'r-', label="Validation R2 Score")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("R2 Score Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the R2 score plot in the same directory
    r2_plot_path = os.path.join(loss_dir_path, f"r2_plot_{latest_index}.png")
    plt.savefig(r2_plot_path)
    plt.close()
    print(f"R2 Score plot saved to {r2_plot_path}")

if __name__ == "__main__":
    plot_latest_training_loss()
