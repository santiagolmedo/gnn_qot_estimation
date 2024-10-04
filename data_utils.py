import os
import pickle
import networkx as nx
import torch
from torch_geometric.utils import from_networkx

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
