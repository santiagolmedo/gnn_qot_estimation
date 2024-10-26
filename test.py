import torch
from torch_geometric.loader import DataLoader
from data_utils import load_topological_graphs_from_pickle
from models import TopologicalGNN
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import numpy as np
from constants import TARGET_RANGES


def min_max_descale(scaled_value, min_value, max_value):
    return scaled_value * (max_value - min_value) + min_value


if __name__ == "__main__":
    # Load the data and FEATURES
    data_list, _ = load_topological_graphs_from_pickle()

    # Split the data into training and testing sets
    test_data = data_list[int(len(data_list) * 0.8) :]
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load the saved model state and parameters from the last training session
    file_name = f"model_{len(os.listdir('models')) - 1}"
    model_path = f"models/{file_name}.pth"
    checkpoint = torch.load(model_path)
    model_params = checkpoint["model_params"]

    # Instantiate the model using the saved parameters
    model = TopologicalGNN(
        num_nodes=model_params["num_nodes"],
        hidden_channels=model_params["hidden_channels"],
        out_channels=model_params["output_dim"],
        edge_dim=model_params["edge_dim"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the model state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {model_path}")

    # Evaluate the model
    model.eval()
    test_loss = 0
    y_true_list = []
    y_pred_list = []

    criterion = torch.nn.SmoothL1Loss()  # Usamos SmoothL1Loss para la evaluación

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            y = data.y.view(-1, model_params["output_dim"])
            loss = criterion(out, y)
            test_loss += loss.item() * data.num_graphs

            y_true_list.append(y.cpu())
            y_pred_list.append(out.cpu())

    test_loss /= len(test_loader.dataset)
    print(f"Test SmoothL1 Loss: {test_loss:.4f}")

    # Convert y_true and y_pred to numpy arrays
    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    # Descale the outputs
    output_keys = ["osnr", "snr", "ber"]
    y_true_descaled = []
    y_pred_descaled = []
    for i, key in enumerate(output_keys):
        min_val = TARGET_RANGES[key]["min"]
        max_val = TARGET_RANGES[key]["max"]
        y_true_descaled.append(min_max_descale(y_true[:, i], min_val, max_val))
        y_pred_descaled.append(min_max_descale(y_pred[:, i], min_val, max_val))

    y_true = np.column_stack(y_true_descaled)
    y_pred = np.column_stack(y_pred_descaled)

    # Compute R2 per output
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    print(f"R2 Score per output: {r2}")

    # Define the names of the outputs
    output_names = ["OSNR", "SNR", "BER"]

    # Create the results folder
    results_folder = (
        f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_name}"
    )
    os.makedirs(results_folder, exist_ok=True)

    # Save the evaluation metrics to a JSON file
    results = {}
    for i in range(model_params["output_dim"]):
        results[output_names[i]] = {
            "R2": float(r2[i]),
            "Test_SmoothL1_Loss": float(test_loss),
        }

    with open(os.path.join(results_folder, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plot the results and save each to the folder
    for i in range(model_params["output_dim"]):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        if i < len(output_names):
            plt.title(output_names[i])
            filename = f"{output_names[i]}_results.png"
        else:
            plt.title(f"Output {i + 1}")
            filename = f"Output_{i + 1}_results.png"
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, filename))
        plt.close()
