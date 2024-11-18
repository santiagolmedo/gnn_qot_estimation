import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
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

    # Split the dataset into training, validation, and testing sets without randomization
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len

    train_dataset_full = Subset(dataset, range(0, train_len))
    val_dataset = Subset(dataset, range(train_len, train_len + val_len))
    test_dataset = Subset(dataset, range(train_len + val_len, total_len))

    batch_size = 2048
    num_workers = 4
    num_epochs = 100
    loss_history = []
    val_loss_history = []
    r2_history = []
    val_r2_history = []
    skipped_graphs = 0
    patience = 30
    best_val_r2 = -np.inf
    patience_counter = 0

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

    # Define the optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    criterion = torch.nn.SmoothL1Loss()

    # Create DataLoaders
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Divide the training dataset into chunks (without randomization)
    num_chunks = int(1 / 0.20)  # Using 20% of data per epoch
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
        skipped_graphs_epoch = 0

        if epoch == 0:
            log_message(
                f"Training model with {len(train_dataset)} samples and validating with {len(val_dataset)} samples."
            )

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            try:
                out, lut_batch = model(
                    data
                )  # Output shape: [num_lut_nodes, output_dim]
            except ValueError as e:
                skipped_graphs += data.num_graphs
                skipped_graphs_epoch += data.num_graphs
                continue

            y = data.y.to(device)  # Shape: [batch_size, output_dim]
            y = y[lut_batch]  # Select y for graphs with LUT nodes

            if y.size(0) == 0:
                skipped_graphs += data.num_graphs
                skipped_graphs_epoch += data.num_graphs
                continue

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

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
                try:
                    out, lut_batch = model(data)
                except ValueError:
                    continue

                y = data.y.to(device)
                y = y[lut_batch]

                if y.size(0) == 0:
                    continue

                loss = criterion(out, y)
                val_total_loss += loss.item() * y.size(0)

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
        log_message(
            f"Skipped {skipped_graphs_epoch} graphs in this epoch due to missing LUT nodes."
        )

        # Check for early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_message("Early stopping triggered.")
                break

        # Step the scheduler
        scheduler.step()

    # Save the final model
    root_dir = "lightpath_training/models"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_name = f"model_{len(os.listdir(root_dir))}.pth"
    model_path = os.path.join(root_dir, file_name)

    log_message(f"Total skipped {skipped_graphs} graphs due to missing LUT nodes.")

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

    # Plot the loss and R2 score history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(r2_history, label="Training R2 Score")
    plt.plot(val_r2_history, label="Validation R2 Score")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("R2 Score Over Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"lightpath_training/loss_{file_name}.png")
