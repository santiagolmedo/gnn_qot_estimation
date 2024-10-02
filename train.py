import networkx as nx
import xarray as xr
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import os
import pickle

DIRECTORY_FOR_GRAPHS = "networkx_graphs"


# Read every graph from the directory and return a list with all of them
def load_graphs_from_pickle(directory):
    data_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Cargar el grafo desde el archivo pickle
        with open(filepath, "rb") as f:
            G = pickle.load(f)

        # Convertir los IDs de los nodos a enteros consecutivos
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")

        # Convertir los atributos de las aristas a flotantes
        for u, v, attr in G.edges(data=True):
            for key, value in attr.items():
                attr[key] = float(value)

        # Convertir el grafo de NetworkX a un objeto Data de PyTorch Geometric
        data = from_networkx(G)

        # Asignar características de nodos (x)
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, 1))  # Características ficticias

        # Extraer las etiquetas del grafo
        labels = G.graph.get("labels", {})
        y = torch.tensor(
            [labels["osnr"], labels["snr"], labels["ber"]], dtype=torch.float
        )
        data.y = y

        # Añadir el objeto Data a la lista
        data_list.append(data)
    return data_list

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
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
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    test_loss = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        test_loss += criterion(out, data.y).item()

    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss}")
