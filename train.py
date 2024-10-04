import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import r2_score
import os
import pickle
import pdb

DIRECTORY_FOR_GRAPHS = "networkx_graphs"


# Read every graph from the directory and return a list with all of them
def load_graphs_from_pickle(directory):
    data_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Cargar el grafo desde el archivo pickle
        with open(filepath, "rb") as f:
            G = pickle.load(f)

        # Transform the node IDs to consecutive integers
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")

        # Transform the edge attributes to floats
        for u, v, attr in G.edges(data=True):
            for key, value in attr.items():
                attr[key] = float(value)

        # Transform the graph to a PyTorch Geometric Data object
        data = from_networkx(G)

        # Assign node features (x)
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, 1))

        # y = [osnr, snr, ber]
        labels = G.graph.get("labels", {})
        y = torch.tensor(
            [labels["osnr"], labels["snr"], labels["ber"]], dtype=torch.float
        )
        data.y = y

        # Add the Data object to the list
        data_list.append(data)
    return data_list

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Convolutional layer
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Linear dense layer
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

if __name__ == "__main__":
    data_list = load_graphs_from_pickle(DIRECTORY_FOR_GRAPHS)

    train_data = data_list[: int(len(data_list) * 0.8)]
    test_data = data_list[int(len(data_list) * 0.8) :]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    num_node_features = 1
    hidden_channels = 16
    output_dim = 3

    model = GCN(num_node_features, hidden_channels, output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(100):
        print(f"Epoch {epoch + 1}")
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            y = data.y.view(-1, output_dim)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        y = data.y.view(-1, output_dim)
        test_loss += criterion(out, y).item()

    test_loss /= len(test_loader)

    # Calculate R2 score
    y_true = y.cpu().detach().numpy()
    y_pred = out.cpu().detach().numpy()
    r2 = r2_score(y_true, y_pred)
    print(f"R2 Score: {r2}")

    print(f"Test Loss: {test_loss}")
