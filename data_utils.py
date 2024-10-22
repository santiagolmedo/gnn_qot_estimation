import os
import pickle
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from constants import FEATURE_RANGES, TARGET_RANGES

def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def load_topological_graphs_from_pickle(directory="networkx_graphs_topological"):
    data_list = []
    FEATURES = set()
    for filename in sorted(os.listdir(directory)):
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
                min_val = FEATURE_RANGES[key]["min"]
                max_val = FEATURE_RANGES[key]["max"]
                attr[key] = min_max_scale(float(value), min_val, max_val)

        # Convert the graph to a PyTorch Geometric Data object
        data = from_networkx(G)

        # Assign node_ids
        data.node_ids = torch.arange(data.num_nodes)

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
        data.x = None  # No features, will use node_embeddings

        # y = [osnr, snr, ber]
        labels = G.graph.get("labels", {})
        y_scaled = []
        for key in ["osnr", "snr", "ber"]:
            min_val = TARGET_RANGES[key]["min"]
            max_val = TARGET_RANGES[key]["max"]
            value = labels.get(key, 0.0)
            scaled_value = min_max_scale(float(value), min_val, max_val)
            y_scaled.append(scaled_value)
        data.y = torch.tensor(y_scaled, dtype=torch.float)

        # Add the Data object to the list
        data_list.append(data)
    # Convert FEATURES to a sorted list for consistent ordering
    FEATURES = sorted(FEATURES)
    return data_list, FEATURES

def load_lightpath_graphs_from_pickle(directory="networkx_graphs_lightpath"):
    data_list = []
    NODE_FEATURES = set()
    for filename in sorted(os.listdir(directory)):
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
                    min_val = FEATURE_RANGES[key]["min"]
                    max_val = FEATURE_RANGES[key]["max"]
                    attr[key] = min_max_scale(float(value), min_val, max_val)
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
        y_scaled = []
        for key in ["osnr", "snr", "ber"]:
            min_val = TARGET_RANGES[key]["min"]
            max_val = TARGET_RANGES[key]["max"]
            value = labels.get(key, 0.0)
            scaled_value = min_max_scale(float(value), min_val, max_val)
            y_scaled.append(scaled_value)
        data.y = torch.tensor(y_scaled, dtype=torch.float)
        # Add the Data object to the list
        data_list.append(data)

    # Convert NODE_FEATURES to a sorted list for consistent ordering
    NODE_FEATURES = sorted(NODE_FEATURES)
    return data_list, NODE_FEATURES