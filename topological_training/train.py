import torch
from torch_geometric.loader import DataLoader
from data_utils import load_topological_graphs_from_pickle
from topological_training.models import TopologicalGNN
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_list, FEATURES = load_topological_graphs_from_pickle()
    edge_dim = len(FEATURES)

    train_data = data_list[: int(len(data_list) * 0.8)]
    test_data = data_list[int(len(data_list) * 0.8) :]

    train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    num_node_features = 1
    num_nodes = 75
    hidden_channels = 16
    output_dim = 3

    model = TopologicalGNN(
        num_nodes, hidden_channels, output_dim, edge_dim=edge_dim, dropout_p=0.4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.SmoothL1Loss()

    # Train the model
    loss_history = []
    for epoch in range(100):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            y = data.y.view(-1, output_dim)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    if not os.path.exists("topological_training/models"):
        os.makedirs("topological_training/models")
    file_name = f"model_{len(os.listdir('topological_training/models'))}.pth"
    model_path = os.path.join("topological_training/models", file_name)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "num_nodes": num_nodes,
                "hidden_channels": hidden_channels,
                "output_dim": output_dim,
                "edge_dim": edge_dim,
                "FEATURES": FEATURES,
            },
        },
        model_path,
    )
    print("Model saved to", model_path)

    # Plot the loss history
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_{file_name}.png")
