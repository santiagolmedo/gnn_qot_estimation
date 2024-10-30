import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import numpy as np
from data_utils import load_lightpath_graphs_from_pickle
from lightpath_training.models import LightpathGNN

if __name__ == "__main__":
    # Load the data
    data_list, NODE_FEATURES, feature_indices = load_lightpath_graphs_from_pickle()
    num_features = len(NODE_FEATURES)

    # Split the data into training and testing sets
    train_data = data_list[: int(len(data_list) * 0.8)]
    test_data = data_list[int(len(data_list) * 0.8) :]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Model
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

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            try:
                out = model(data)  # Output shape: [output_dim]
            except ValueError as e:
                skipped_graphs += 1
                continue
            y = data.y.to(device)  # Shape: [output_dim]

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            y_true.append(y.cpu().detach().numpy())
            y_pred.append(out.cpu().detach().numpy())

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Calculate R2 score
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
        r2_history.append(r2)

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, R2 Score: {r2:.4f}")

    # Save the model
    root_dir = "lightpath_training/models"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_name = f"model_{len(os.listdir(root_dir))}.pth"
    model_path = os.path.join(root_dir, file_name)

    print(f"Skipped {skipped_graphs} graphs due to missing LUT nodes.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "in_channels": num_features,
                "hidden_channels": hidden_channels,
                "output_dim": output_dim,
                "NODE_FEATURES": NODE_FEATURES,
            },
        },
        model_path,
    )
    print("Model saved to", model_path)

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
