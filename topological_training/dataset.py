import os
import torch
from torch.utils.data import Dataset
import networkx as nx
import pickle
from torch_geometric.utils import from_networkx
from constants import FEATURE_RANGES, TARGET_RANGES

def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class TopologicalDataset(Dataset):
    def __init__(self, directory="networkx_graphs_topological", features=None):
        self.directory = directory
        self.file_list = sorted(
            [f for f in os.listdir(directory) if f.endswith(".gpickle")]
        )
        self.N = len(self.file_list)

        # If features are provided, use them
        self.FEATURES = features

        # If not, process a subset of files to collect them
        if self.FEATURES is None:
            self.FEATURES = set()
            counter = 0
            # Process a subset of files to gather all possible edge features
            for filename in self.file_list[:100]:  # Adjust the number as needed
                filepath = os.path.join(self.directory, filename)
                with open(filepath, "rb") as f:
                    G = pickle.load(f)
                # Collect all edge attribute names
                for _, _, attr in G.edges(data=True):
                    self.FEATURES.update(attr.keys())
                counter += 1
            # Convert FEATURES to a sorted list for consistent ordering
            self.FEATURES = sorted(self.FEATURES)
        self.edge_dim = len(self.FEATURES)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.directory, filename)
        # Load the graph from the pickle file
        try:
            with open(filepath, "rb") as f:
                G = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"Error loading {filepath}: {e}")
            return self.__getitem__((idx + 1) % len(self.file_list))

        # Transform the node IDs to consecutive integers
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")

        # Remove the 'original_id' attribute
        for node, attr in G.nodes(data=True):
            if "original_id" in attr:
                del attr["original_id"]

        # Convert edge attributes to floats and normalize them
        for u, v, attr in G.edges(data=True):
            for key, value in attr.items():
                try:
                    min_val = FEATURE_RANGES[key]["min"]
                    max_val = FEATURE_RANGES[key]["max"]
                    attr[key] = min_max_scale(float(value), min_val, max_val)
                except (ValueError, KeyError):
                    pass  # Handle missing keys or invalid values

        # Convert the graph to a PyTorch Geometric Data object
        data = from_networkx(G)

        # Assign node_ids
        data.node_ids = torch.arange(data.num_nodes)

        num_edges = data.edge_index.size(1)
        edge_attrs = []

        # Create a dictionary to map edge indices to their attributes
        edge_attr_dict = {}
        for u, v, attr in G.edges(data=True):
            # Ensure consistent order of FEATURES
            edge_attr_dict[(u, v)] = [
                attr.get(key, 0.0) for key in self.FEATURES
                if isinstance(attr.get(key, 0.0), (int, float))
            ]

        # Align the attributes with data.edge_index
        for i in range(num_edges):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            # Get the edge attributes for the pair (u, v) or (v, u)
            attr = edge_attr_dict.get((u, v)) or edge_attr_dict.get((v, u))
            if attr is None:
                # If edge attributes are missing, use zeros
                attr = [0.0] * len(self.FEATURES)
            edge_attrs.append(attr)

        data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # Assign node features (x)
        data.x = None  # No node features, will use node_embeddings

        # y = [osnr, snr, ber]
        labels = G.graph.get("labels", {})
        y_scaled = []
        for key in ["osnr", "snr", "ber"]:
            try:
                min_val = TARGET_RANGES[key]["min"]
                max_val = TARGET_RANGES[key]["max"]
                value = labels.get(key, 0.0)
                scaled_value = min_max_scale(float(value), min_val, max_val)
                y_scaled.append(scaled_value)
            except (ValueError, KeyError):
                y_scaled.append(0.0)  # Handle missing labels
        data.y = torch.tensor(y_scaled, dtype=torch.float)

        return data
