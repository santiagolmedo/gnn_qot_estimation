import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightpath_training.dataset import LightpathDataset
from lightpath_training.models import LightpathGNN
from constants import TARGET_RANGES

def min_max_descale(scaled_value, min_value, max_value):
    return scaled_value * (max_value - min_value) + min_value

if __name__ == "__main__":
    # Load the dataset
    dataset = LightpathDataset()
    num_features = len(dataset.node_features)
    feature_indices = dataset.feature_indices

    # Split the dataset into training, validation, and testing sets without randomization
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len

    # The test dataset is the last part of the dataset
    test_dataset = Subset(dataset, range(train_len + val_len, total_len))

    batch_size = 512

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the saved model state and parameters from the last training session
    models_dir = "lightpath_training/models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError("No saved models found in the 'models' directory.")

    # Find the latest model based on the index
    model_indices = [int(f.split("_")[1].split(".")[0]) for f in model_files]
    latest_model_index = max(model_indices)
    model_file_name = f"model_{latest_model_index}.pth"
    model_path = os.path.join(models_dir, model_file_name)
    print(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path)
    model_params = checkpoint["model_params"]

    # Initialize the model
    model = LightpathGNN(
        in_channels=model_params["in_channels"],
        hidden_channels=model_params["hidden_channels"],
        output_dim=model_params["output_dim"],
        is_lut_index=feature_indices["is_lut"],
        dropout_p=0.0,  # Dropout is not used during evaluation
    )

    model.load_state_dict(checkpoint["model_state_dict"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true_list = []
    y_pred_list = []

    criterion = torch.nn.SmoothL1Loss()

    skipped_graphs = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            try:
                out, lut_batch = model(data)
            except ValueError as e:
                skipped_graphs += data.num_graphs
                continue

            y = data.y.to(device)
            y = y[lut_batch]

            if y.size(0) == 0:
                skipped_graphs += data.num_graphs
                continue

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(out.cpu().numpy())

    if len(y_true_list) == 0 or len(y_pred_list) == 0:
        raise ValueError("No valid data to evaluate after skipping graphs.")

    # Transform y_true and y_pred to numpy arrays
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # Unscale the outputs
    output_keys = ["osnr", "snr", "ber"]
    y_true_descaled = []
    y_pred_descaled = []
    for i, key in enumerate(output_keys):
        min_val = TARGET_RANGES[key]["min"]
        max_val = TARGET_RANGES[key]["max"]
        y_true_descaled.append(min_max_descale(y_true[:, i], min_val, max_val))
        y_pred_descaled.append(min_max_descale(y_pred[:, i], min_val, max_val))

    y_true_descaled = np.column_stack(y_true_descaled)
    y_pred_descaled = np.column_stack(y_pred_descaled)

    # Calculate R2 Score and MSE
    r2 = r2_score(y_true_descaled, y_pred_descaled, multioutput="raw_values")
    mse = mean_squared_error(y_true_descaled, y_pred_descaled, multioutput="raw_values")
    print(f"Test R2 Score per output: {r2}")
    print(f"Test MSE per output: {mse}")
    print(f"Total skipped graphs during testing: {skipped_graphs}")

    # Save the results
    output_names = ["OSNR", "SNR", "BER"]

    # Create the results folder
    results_folder = f"lightpath_training/results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_folder, exist_ok=True)

    # Save the evaluation metrics to a JSON file
    results = {}
    for i in range(model_params["output_dim"]):
        results[output_names[i]] = {
            "R2": float(r2[i]),
            "Test_MSE": float(mse[i]),
        }

    with open(os.path.join(results_folder, "results.json"), "w") as f:
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