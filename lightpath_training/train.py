import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from lightpath_training.dataset import LightpathDataset
from lightpath_training.models import LightpathGNN

def log_message(*args):
    message = " ".join(map(str, args))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp} - {message}"
    print(full_message)
    with open(
        os.path.join(os.path.dirname(__file__), "model_logger.txt"),
        "a",
    ) as logger:
        logger.write(full_message + "\n")

if __name__ == "__main__":
    # Load the dataset
    dataset = LightpathDataset()
    num_features = len(dataset.node_features)
    feature_indices = dataset.feature_indices

    # Split the dataset into training and testing sets
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    test_len = total_len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])

    batch_size = 2048
    num_workers = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Define the model
    hidden_channels = 32
    output_dim = 3  # OSNR, SNR, BER

    model = LightpathGNN(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        output_dim=output_dim,
        is_lut_index=feature_indices["is_lut"],
        dropout_p=0.5,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.SmoothL1Loss()

    num_epochs = 50
    loss_history = []
    r2_history = []
    skipped_graphs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            try:
                out, lut_batch = model(data)  # Output shape: [num_lut_nodes, output_dim]
            except ValueError as e:
                skipped_graphs += data.num_graphs
                continue

            y = data.y.to(device)  # Shape: [batch_size, output_dim]
            y = y[lut_batch]  # Select y for graphs with LUT nodes

            if y.size(0) == 0:
                skipped_graphs += data.num_graphs
                continue

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

            y_true.append(y.cpu().detach().numpy())
            y_pred.append(out.cpu().detach().numpy())
            if batch_idx % 10:
                log_message(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataset)
        loss_history.append(avg_loss)

        # Calculate R2 score
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
        r2_history.append(r2)

        log_message(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, R2 Score: {r2:.4f}")

    # Save the model
    root_dir = "lightpath_training/models"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_name = f"model_{len(os.listdir(root_dir))}.pth"
    model_path = os.path.join(root_dir, file_name)

    log_message(f"Skipped {skipped_graphs} graphs due to missing LUT nodes.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "in_channels": num_features,
                "hidden_channels": hidden_channels,
                "output_dim": output_dim,
                "NODE_FEATURES": dataset.node_features,
                "feature_indices": feature_indices,
            },
        },
        model_path,
    )
    log_message("Model saved to", model_path)

    # Plot the loss history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(r2_history, label="Training R2 Score")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Training R2 Score Over Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"lightpath_training/loss_{file_name}.png")