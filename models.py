import torch
from torch_geometric.nn import GCNConv, global_mean_pool

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
