import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from data_utils import load_lightpath_graphs_from_pickle
from lightpath_training.models import LightpathGNN
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from constants import TARGET_RANGES


def min_max_descale(scaled_value, min_value, max_value):
    return scaled_value * (max_value - min_value) + min_value


if __name__ == "__main__":
    # Load the data
    data_list, NODE_FEATURES, feature_indices = load_lightpath_graphs_from_pickle()
    num_features = len(NODE_FEATURES)

    # Split the data into training and testing sets
    train_data = data_list[: int(len(data_list) * 0.8)]
    test_data = data_list[int(len(data_list) * 0.8) :]

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Load the saved model state and parameters from the last training session
    file_name = f"model_{len(os.listdir('lightpath_training/models')) - 1}"
    model_path = f"lightpath_training/models/{file_name}.pth"
    checkpoint = torch.load(model_path)
    model_params = checkpoint["model_params"]

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

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            try:
                out = model(data)
            except ValueError as e:
                print(f"Skipping graph due to error: {e}")
                continue

            y = data.y.to(device)

            y_true_list.append(y.cpu())
            y_pred_list.append(out.cpu())

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

    # R2 Score y MSE
    r2 = r2_score(y_true_descaled, y_pred_descaled, multioutput="raw_values")
    mse = mean_squared_error(y_true_descaled, y_pred_descaled, multioutput="raw_values")
    print(f"Test R2 Score per output: {r2}")
    print(f"Test MSE per output: {mse}")

    # Save the results
    output_names = ["OSNR", "SNR", "BER"]

    # Create the results folder
    results_folder = (
        f"lightpath_training/results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
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

    # Plot the results and save each to the folder
    for i in range(model_params["output_dim"]):
        plt.figure()
        plt.scatter(y_true_descaled[:, i], y_pred_descaled[:, i], alpha=0.5)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(output_names[i])
        plt.grid(True)
        plt.tight_layout()
        filename = f"{output_names[i]}_results.png"
        plt.savefig(os.path.join(results_folder, filename))
        plt.close()

    print(f"Results saved to {results_folder}")
