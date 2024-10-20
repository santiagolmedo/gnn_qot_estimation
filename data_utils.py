import os
import pickle
import networkx as nx
import torch
from torch_geometric.utils import from_networkx

# Read every graph from the directory and return a list with all of them
def load_topological_graphs_from_pickle(directory):
    data_list = []
    FEATURES = set()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Load the graph from the pickle file
        with open(filepath, "rb") as f:
            G = pickle.load(f)

        # Transform the node IDs to consecutive integers
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")

        # Collect all edge attribute names
        for _, _, attr in G.edges(data=True):
            FEATURES.update(attr.keys())

        # Convert edge attributes to floats
        for u, v, attr in G.edges(data=True):
            for key, value in attr.items():
                attr[key] = float(value)

        # Convert the graph to a PyTorch Geometric Data object
        data = from_networkx(G)

        num_edges = data.edge_index.size(1)
        edge_attrs = []

        # Create a dictionary to map node pairs to their attributes
        edge_attr_dict = {}
        for u, v, attr in G.edges(data=True):
            # Ensure consistent order of FEATURES
            edge_attr_dict[(u, v)] = [attr.get(key, 0.0) for key in FEATURES]

        # Align the attributes with data.edge_index
        for i in range(num_edges):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            # Get the edge attributes for the pair (u, v) or (v, u)
            attr = edge_attr_dict.get((u, v)) or edge_attr_dict.get((v, u))
            if attr is None:
                raise ValueError(
                    f"Edge ({u}, {v}) or ({v}, {u}) not found in edge_attr_dict"
                )
            edge_attrs.append(attr)

        data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # Assign node features (x)
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, 1))  # Placeholder for the node features

        # y = [osnr, snr, ber]
        labels = G.graph.get("labels", {})
        y = torch.tensor(
            [labels["osnr"], labels["snr"], labels["ber"]], dtype=torch.float
        )
        data.y = y

        # Add the Data object to the list
        data_list.append(data)
    # Convert FEATURES to a sorted list for consistent ordering
    FEATURES = sorted(FEATURES)
    return data_list, FEATURES

def load_lightpath_graphs_from_pickle(directory):
    data_list = []
    NODE_FEATURES = set()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Load the graph from the pickle file
        with open(filepath, "rb") as f:
            G = pickle.load(f)

        # Transform the node IDs to consecutive integers
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")

        # Collect all node attribute names
        for _, attr in G.nodes(data=True):
            NODE_FEATURES.update(attr.keys())

        # Convert node attributes to floats
        for node, attr in G.nodes(data=True):
            for key, value in attr.items():
                try:
                    attr[key] = float(value)
                except ValueError:
                    pass

        # Convert the graph to a PyTorch Geometric Data object
        data = from_networkx(G)

        num_nodes = data.num_nodes
        node_attrs = []

        # Create a dictionary to map node IDs to their attributes
        node_attr_dict = {}
        for node, attr in G.nodes(data=True):
            node_attr_dict[node] = [
                attr.get(key, 0.0) for key in sorted(NODE_FEATURES) if isinstance(attr.get(key, 0.0), (int, float))
            ]

        # Align the attributes with data.x
        for i in range(num_nodes):
            node_id = i  # Since node labels are now consecutive integers starting from 0
            attr = node_attr_dict.get(node_id)
            if attr is None:
                raise ValueError(f"Node {node_id} not found in node_attr_dict")
            node_attrs.append(attr)

        data.x = torch.tensor(node_attrs, dtype=torch.float)

        data.edge_attr

        # y = [osnr, snr, ber]
        labels = G.graph.get("labels", {})
        y = torch.tensor(
            [labels["osnr"], labels["snr"], labels["ber"]], dtype=torch.float
        )
        data.y = y
        # Add the Data object to the list
        data_list.append(data)

    # Convert NODE_FEATURES to a sorted list for consistent ordering
    NODE_FEATURES = sorted(NODE_FEATURES)
    return data_list, NODE_FEATURES