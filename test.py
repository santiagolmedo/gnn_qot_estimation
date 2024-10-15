import torch
from torch_geometric.loader import DataLoader
from data_utils import load_graphs_from_pickle
from models import GCN
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import numpy as np

DIRECTORY_FOR_GRAPHS = "networkx_graphs"

def mape_loss(output, target):
    epsilon = 1e-8
    return torch.mean(torch.abs((target - output) / (target + epsilon))) * 100


if __name__ == "__main__":
    # Load the data and FEATURES
    data_list, _ = load_graphs_from_pickle(DIRECTORY_FOR_GRAPHS)

    # Split the data into training and testing sets
    test_data = data_list[int(len(data_list) * 0.8) :]
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Load the saved model state and parameters from the last training session
    file_name = f"model_{len(os.listdir('models')) - 1}"
    model_path = f"models/{file_name}.pth"
    checkpoint = torch.load(model_path)
    model_params = checkpoint["model_params"]

    # Instantiate the model using the saved parameters
    model = GCN(
        in_channels=model_params["num_node_features"],
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
    test_mape_loss = 0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            y = data.y.view(-1, model_params["output_dim"])
            mape = mape_loss(out, y)
            mse = torch.nn.functional.mse_loss(out, y)
            test_mape_loss += mape.item() * data.num_graphs

            y_true_list.append(y.cpu())
            y_pred_list.append(out.cpu())

    test_mape_loss /= len(test_loader.dataset)
    print(f"Test MAPE Loss: {test_mape_loss:.4f}")

    # Convert y_true and y_pred to numpy arrays
    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    epsilon = 1e-8
    mape_per_output = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)), axis=0) * 100

    # Compute R2 per output
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')

    print(f"MAPE per output: {mape_per_output}")
    print(f"R2 Score per output: {r2}")

    # Define the names of the outputs
    output_names = ["OSNR", "SNR", "BER"]

    # Create the results folder
    results_folder = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_name}"
    os.makedirs(results_folder, exist_ok=True)

    # Save the evaluation metrics to a JSON file
    results = {}
    for i in range(model_params["output_dim"]):
        results[output_names[i]] = {
            "R2": float(r2[i]),
            "MAPE": float(mape_per_output[i]),
            "Test_MAPE_Loss": float(test_mape_loss)
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
