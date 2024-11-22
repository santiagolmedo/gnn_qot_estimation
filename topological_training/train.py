import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import r2_score
import os
import numpy as np
import time
from topological_training.dataset import TopologicalDataset
from topological_training.models import TopologicalGNN
import json

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
    dataset = TopologicalDataset(directory="networkx_graphs_topological")
    edge_dim = dataset.edge_dim

    # Split the dataset into training, validation, and testing sets without randomization
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len

    train_dataset_full = Subset(dataset, range(0, train_len))
    val_dataset = Subset(dataset, range(train_len, train_len + val_len))
    test_dataset = Subset(dataset, range(train_len + val_len, total_len))

    batch_size = 512
    num_workers = 4
    num_epochs = 35
    loss_history = []
    val_loss_history = []
    r2_history = []
    val_r2_history = []
    patience = 10
    best_val_r2 = -np.inf
    patience_counter = 0

    # Define the model
    num_nodes = 75  # Adjust according to your data
    hidden_channels = 16
    output_dim = 3  # OSNR, SNR, BER

    model = TopologicalGNN(
        num_nodes=num_nodes,
        hidden_channels=hidden_channels,
        out_channels=output_dim,
        edge_dim=edge_dim,
        dropout_p=0.5,
    )

    device = torch.device("cpu")  # Training on CPU
    model = model.to(device)

    # Define the optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    criterion = torch.nn.SmoothL1Loss()

    # Create DataLoaders
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Divide the training dataset into chunks (without randomization)
    num_chunks = int(1 / 0.10)  # Using 10% of data per epoch
    chunk_size = train_len // num_chunks
    train_indices = list(range(train_len))

    for epoch in range(num_epochs):
        # Determine the indices for this epoch's chunk
        start_idx = (epoch % num_chunks) * chunk_size
        end_idx = start_idx + chunk_size
        if end_idx > train_len:
            end_idx = train_len

        # Create a Subset for this epoch
        train_indices_epoch = train_indices[start_idx:end_idx]
        train_dataset = Subset(train_dataset_full, train_indices_epoch)

        # Create DataLoader for this epoch
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        if epoch == 0:
            log_message(
                f"Training model with {len(train_dataset)} samples and validating with {len(val_dataset)} samples."
            )

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)  # Output shape: [batch_size, output_dim]
            y = data.y.view(-1, output_dim).to(device)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

            y_true.append(y.cpu().detach().numpy())
            y_pred.append(out.cpu().detach().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

        # Calculate R2 score
        if len(y_true) > 0 and len(y_pred) > 0:
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
        else:
            r2 = float("nan")
        r2_history.append(r2)

        # Validation
        model.eval()
        val_total_loss = 0
        val_y_true = []
        val_y_pred = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                y = data.y.view(-1, output_dim).to(device)

                loss = criterion(out, y)
                val_total_loss += loss.item() * data.num_graphs

                val_y_true.append(y.cpu().numpy())
                val_y_pred.append(out.cpu().numpy())

        avg_val_loss = val_total_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)

        # Calculate validation R2 score
        if len(val_y_true) > 0 and len(val_y_pred) > 0:
            val_y_true = np.vstack(val_y_true)
            val_y_pred = np.vstack(val_y_pred)
            val_r2 = r2_score(val_y_true, val_y_pred, multioutput="uniform_average")
        else:
            val_r2 = float("nan")
        val_r2_history.append(val_r2)

        log_message(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, R2 Score: {r2:.4f}, Val Loss: {avg_val_loss:.4f}, Val R2 Score: {val_r2:.4f}"
        )

        # Check for early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_message("Early stopping triggered.")
                break

        # Step the scheduler
        scheduler.step()

    # Save the model
    root_dir = "topological_training/models"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    existing_models = [name for name in os.listdir(root_dir) if name.startswith("model_")]
    if existing_models:
        existing_indices = [int(name.split("_")[1].split(".")[0]) for name in existing_models]
        model_index = max(existing_indices) + 1
    else:
        model_index = 0

    file_name = f"model_{model_index}.pth"
    model_path = os.path.join(root_dir, file_name)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "num_nodes": num_nodes,
                "hidden_channels": hidden_channels,
                "output_dim": output_dim,
                "edge_dim": edge_dim,
                "FEATURES": dataset.FEATURES,
            },
        },
        model_path,
    )
    log_message("Model saved to", model_path)

    # Save the loss and metrics
    loss_dir = f"lightpath_training/loss_training_{model_index}"
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    with open(os.path.join(loss_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)

    with open(os.path.join(loss_dir, "val_loss_history.json"), "w") as f:
        json.dump(val_loss_history, f)

    with open(os.path.join(loss_dir, "r2_history.json"), "w") as f:
        json.dump(r2_history, f)

    with open(os.path.join(loss_dir, "val_r2_history.json"), "w") as f:
        json.dump(val_r2_history, f)
