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
        x, edge_index = data.x, data.edge_index

        # GNN layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)

        # Extract LUT embedding
        lut_mask = data.x[:, self.is_lut_index] == 1.0
        lut_embedding = x[lut_mask]
        if lut_embedding.shape[0] == 0:
            raise ValueError("No LUT node found in the graph.")

        # Prediction
        out = self.mlp(lut_embedding)
        out = out.squeeze(0)  # Shape: [output_dim]
        return out
