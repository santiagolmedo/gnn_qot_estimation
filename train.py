import torch
from torch_geometric.loader import DataLoader
from data_utils import load_graphs_from_pickle
from models import GCN

DIRECTORY_FOR_GRAPHS = "networkx_graphs"
MODEL_PATH = "model.pth"

if __name__ == "__main__":
    data_list, FEATURES = load_graphs_from_pickle(DIRECTORY_FOR_GRAPHS)
    edge_dim = len(FEATURES)

    train_data = data_list[: int(len(data_list) * 0.8)]
    test_data = data_list[int(len(data_list) * 0.8) :]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    num_node_features = 1
    hidden_channels = 16
    output_dim = 3

    model = GCN(num_node_features, hidden_channels, output_dim, edge_dim=edge_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Train the model
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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "num_node_features": num_node_features,
                "hidden_channels": hidden_channels,
                "output_dim": output_dim,
                "edge_dim": edge_dim,
                "FEATURES": FEATURES,
            },
        },
        MODEL_PATH,
    )
    print("Model saved to", MODEL_PATH)
