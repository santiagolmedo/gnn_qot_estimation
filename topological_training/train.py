import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from topological_training.dataset import TopologicalDataset
from topological_training.models import TopologicalGNN

def log_message(*args):
    message = " ".join(map(str, args))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp} - {message}"
    print(full_message)
    with open("model_logger.txt", "a") as logger:
        logger.write(full_message + "\n")

if __name__ == "__main__":
    # Load the dataset
    dataset = TopologicalDataset(directory='networkx_graphs_topological')
    edge_dim = dataset.edge_dim

    # Split the dataset into training and testing sets
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    test_len = total_len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, test_len]
    )

    batch_size = 512
    num_workers = 4
    num_epochs = 50
    loss_history = []
    r2_history = []

    # Define the model
    num_nodes = 75  # Adjust according to your data
    hidden_channels = 16
    output_dim = 3  # OSNR, SNR, BER

    model = TopologicalGNN(
        num_nodes=num_nodes,
        hidden_channels=hidden_channels,
        out_channels=output_dim,
        edge_dim=edge_dim,
        dropout_p=0.4,
    )

    device = torch.device("cpu")  # Training on CPU
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        # Reorder indices in each epoch to use 20% of the training data
        indices = list(range(train_len))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * train_len))
        train_idx = indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)

        # Create DataLoader with the new sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )

        # Log DataLoader length and model architecture in the first epoch
        if epoch == 0:
            log_message("Train DataLoader length:", len(train_loader))
            log_message("Model architecture:", model)

        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

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

            # Log only every fifth of the data per epoch
            if (batch_idx + 1) % (len(train_loader) // 5 + 1) == 0:
                log_message(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_idx)
        loss_history.append(avg_loss)

        # Calculate R2 score
        if len(y_true) > 0 and len(y_pred) > 0:
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
        else:
            r2 = float("nan")
        r2_history.append(r2)

        log_message(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, R2 Score: {r2:.4f}")

    # Save the model
    root_dir = "topological_training/models"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_name = f"model_{len(os.listdir(root_dir))}.pth"
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
    plt.savefig(f"topological_training/loss_{file_name}.png")
