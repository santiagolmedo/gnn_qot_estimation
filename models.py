import torch
from torch_geometric.nn import global_mean_pool, TransformerConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        # Convolutional layer with edge attributes
        self.conv1 = TransformerConv(
            in_channels, hidden_channels, edge_dim=edge_dim
        )
        # Linear dense layer
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x
