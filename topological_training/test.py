# topological_training/test.py

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import json
from datetime import datetime
from topological_training.dataset import TopologicalDataset
from topological_training.models import TopologicalGNN
from constants import TARGET_RANGES

def min_max_descale(scaled_value, min_value, max_value):
    """Descale the scaled value using min-max scaling."""
    return scaled_value * (max_value - min_value) + min_value

if __name__ == "__main__":
    # Load the dataset
    dataset = TopologicalDataset(directory="networkx_graphs_topological")
    edge_dim = dataset.edge_dim

    # Split the dataset into training, validation, and testing sets without randomization
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len

    # Create subsets for training, validation, and testing
    train_dataset_full = Subset(dataset, range(0, train_len))
    val_dataset = Subset(dataset, range(train_len, train_len + val_len))
    test_dataset = Subset(dataset, range(train_len + val_len, total_len))

    # Create DataLoader for the test dataset
    batch_size = 512  # Adjust batch size as needed
    num_workers = 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load the saved model state and parameters from the last training session
    models_dir = "topological_training/models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError("No saved models found in the 'models' directory.")

    # Find the latest model based on the index
    model_indices = [int(f.split("_")[1].split(".")[0]) for f in model_files]
    latest_model_index = max(model_indices)
    model_file_name = f"model_{latest_model_index}.pth"
    model_path = os.path.join(models_dir, model_file_name)
    print(f"Loading model from {model_path}")

    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    model_params = checkpoint["model_params"]

    # Instantiate the model using the saved parameters
    model = TopologicalGNN(
        num_nodes=model_params["num_nodes"],
        hidden_channels=model_params["hidden_channels"],
        out_channels=model_params["output_dim"],
        edge_dim=model_params["edge_dim"],
        dropout_p=0.0,  # Dropout is not used during evaluation
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the model state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {model_path}")

    # Evaluate the model
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)  # Output shape: [batch_size, output_dim]
            y = data.y.view(-1, model_params["output_dim"]).to(device)

            # Collect true and predicted values
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(out.cpu().numpy())

    # Concatenate the lists to form arrays
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # Descale the outputs to their original scale
    output_keys = ["osnr", "snr", "ber"]
    y_true_descaled = []
    y_pred_descaled = []
    for i, key in enumerate(output_keys):
        min_val = TARGET_RANGES[key]["min"]
        max_val = TARGET_RANGES[key]["max"]
        y_true_descaled.append(min_max_descale(y_true[:, i], min_val, max_val))
        y_pred_descaled.append(min_max_descale(y_pred[:, i], min_val, max_val))

    # Combine the descaled outputs into arrays
    y_true_descaled = np.column_stack(y_true_descaled)
    y_pred_descaled = np.column_stack(y_pred_descaled)

    # Compute R2 Score and MSE per output
    r2 = r2_score(y_true_descaled, y_pred_descaled, multioutput="raw_values")
    mse = mean_squared_error(y_true_descaled, y_pred_descaled, multioutput="raw_values")
    print(f"Test R2 Score per output: {r2}")
    print(f"Test MSE per output: {mse}")

    # Define the names of the outputs
    output_names = ["OSNR", "SNR", "BER"]

    # Create the results folder with timestamp and model index
    results_folder = f"topological_training/results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_model_{latest_model_index}"
    os.makedirs(results_folder, exist_ok=True)

    # Save the evaluation metrics to a JSON file
    results = {}
    for i in range(model_params["output_dim"]):
        results[output_names[i]] = {
            "R2": float(r2[i]),
            "Test_MSE": float(mse[i]),
        }

    with open(os.path.join(results_folder, "results_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save the predictions and true values to JSON files
    # Convert arrays to lists for JSON serialization
    y_true_descaled_list = y_true_descaled.tolist()
    y_pred_descaled_list = y_pred_descaled.tolist()

    with open(os.path.join(results_folder, "y_true_descaled.json"), "w") as f:
        json.dump(y_true_descaled_list, f)

    with open(os.path.join(results_folder, "y_pred_descaled.json"), "w") as f:
        json.dump(y_pred_descaled_list, f)

    print(f"Results saved to {results_folder}")
