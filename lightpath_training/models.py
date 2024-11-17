import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch.nn import Linear, Dropout, LeakyReLU


class LightpathGNN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, output_dim, is_lut_index, dropout_p=0.5
    ):
        super().__init__()
        # GNN layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.norm1 = BatchNorm(hidden_channels * 4)

        # MLP for prediction
        self.mlp = torch.nn.Sequential(
            Linear(hidden_channels * 4, hidden_channels),
            LeakyReLU(),
            Dropout(p=dropout_p),
            Linear(hidden_channels, output_dim),
        )

        self.is_lut_index = is_lut_index

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GNN layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)

        # Extract LUT embeddings
        lut_mask = data.x[:, self.is_lut_index] == 1.0
        if not lut_mask.any():
            raise ValueError("No LUT node found in the batch.")

        lut_embedding = x[lut_mask]  # Shape: [num_lut_nodes, hidden_channels * 4]
        lut_batch = batch[lut_mask]  # Batch IDs for LUT nodes

        # Prediction
        out = self.mlp(lut_embedding)  # Shape: [num_lut_nodes, output_dim]

        return out, lut_batch
