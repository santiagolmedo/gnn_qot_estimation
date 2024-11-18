import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TransformerConv, NNConv
from torch.nn import Sequential as Seq, Linear, ReLU, Dropout, LeakyReLU

class TopologicalGNN(torch.nn.Module):
    def __init__(
        self, num_nodes, hidden_channels, out_channels, edge_dim, dropout_p=0.5
    ):
        super().__init__()
        # Node embeddings
        self.node_embeddings = torch.nn.Embedding(num_nodes, hidden_channels)

        # Attention layer
        self.conv1 = TransformerConv(
            hidden_channels, hidden_channels, edge_dim=edge_dim
        )

        # NNConv layer
        nn = Seq(
            Linear(edge_dim, edge_dim * 2),
            ReLU(),
            Linear(edge_dim * 2, hidden_channels * hidden_channels),
        )
        self.conv2 = NNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            nn=nn,
            aggr="mean",
        )

        # MLP for prediction
        self.mlp = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            LeakyReLU(),
            Dropout(p=dropout_p),
            Linear(hidden_channels, out_channels),
        )

        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # If x is None or empty, use node embeddings
        if x is None or x.numel() == 0:
            x = self.node_embeddings(data.node_ids)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        # MLP for prediction
        out = self.mlp(x)
        return out
