import os
import torch
from torch.utils.data import Dataset
import networkx as nx
import pickle
from torch_geometric.utils import from_networkx
from constants import FEATURE_RANGES, TARGET_RANGES

def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class LightpathDataset(Dataset):
    def __init__(self, directory="networkx_graphs_lightpath", node_features=None, feature_indices=None):
        self.directory = directory
        self.file_list = sorted([f for f in os.listdir(directory) if f.endswith(".gpickle")])
        self.N = len(self.file_list)

        # If node_features and feature_indices are provided, use them
        self.node_features = node_features
        self.feature_indices = feature_indices

        # If not, process a subset of files to collect them
        if self.node_features is None or self.feature_indices is None:
            self.node_features = set()
            counter = 0
            # Process a subset of files to gather all possible node features
            for filename in self.file_list[:1000]:  # Adjust the number as needed
                filepath = os.path.join(self.directory, filename)
                with open(filepath, "rb") as f:
                    G = pickle.load(f)
                # Collect all node attribute names
                for _, attr in G.nodes(data=True):
                    self.node_features.update(attr.keys())
                counter += 1
            # Convert node_features to a sorted list for consistent ordering
            self.node_features = sorted(self.node_features)
            self.feature_indices = {key: idx for idx, key in enumerate(self.node_features)}

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
            print(f"Error al cargar {filepath}: {e}")
            return self.__getitem__((idx + 1) % len(self.file_list))

        # Transform the node IDs to consecutive integers
        G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")

        # Remove the 'original_id' attribute
        for node, attr in G.nodes(data=True):
            if "original_id" in attr:
                del attr["original_id"]

        # Convert node attributes to floats
        for node, attr in G.nodes(data=True):
            for key, value in attr.items():
                if key == "is_lut":
                    attr[key] = float(value)  # Keep original value (0.0 or 1.0)
                else:
                    try:
                        min_val = FEATURE_RANGES[key]["min"]
                        max_val = FEATURE_RANGES[key]["max"]
                        attr[key] = min_max_scale(float(value), min_val, max_val)
                    except (ValueError, KeyError):
                        pass  # Handle missing keys or invalid values

        # Convert the graph to a PyTorch Geometric Data object
        data = from_networkx(G)

        num_nodes = data.num_nodes
        node_attrs = []

        # Create a dictionary to map node IDs to their attributes
        node_attr_dict = {}
        for node, attr in G.nodes(data=True):
            node_attr_dict[node] = [
                attr.get(key, 0.0)
                for key in self.node_features
                if isinstance(attr.get(key, 0.0), (int, float))
            ]

        # Align the attributes with data.x
        for i in range(num_nodes):
            node_id = i  # Since node labels are now consecutive integers starting from 0
            attr = node_attr_dict.get(node_id)
            if attr is None:
                raise ValueError(f"Node {node_id} not found in node_attr_dict")
            node_attrs.append(attr)

        data.x = torch.tensor(node_attrs, dtype=torch.float)

        # y = [osnr, snr, ber]
        labels = G.graph.get("labels", {})
        y_scaled = []
        for key in ["osnr", "snr", "ber"]:
            min_val = TARGET_RANGES[key]["min"]
            max_val = TARGET_RANGES[key]["max"]
            value = labels.get(key, 0.0)
            scaled_value = min_max_scale(float(value), min_val, max_val)
            y_scaled.append(scaled_value)
        data.y = torch.tensor(y_scaled, dtype=torch.float).unsqueeze(0)

        return data