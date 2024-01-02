import mlx.core as mx
import mlx.nn as nn

from message_passing import MessagePassing


class GCNLayer(MessagePassing):
    def __init__(self, x_dim, h_dim, bias=True):
        super().__init__(aggr="add")
        
        self.linear = nn.Linear(x_dim, h_dim, bias)

    def __call__(self, x, edge_index, **kwargs):
        x = self.linear(x)
        x = self.propagate(x=x, edge_index=edge_index)

        return x


class GCN(nn.Module):
    def __init__(self, x_dim, h_dim, out_dim, nb_layers=2, dropout=0.5, bias=True):
        super(GCN, self).__init__()

        layer_sizes = [x_dim] + [h_dim] * nb_layers + [out_dim]
        self.gcn_layers = [
            GCNLayer(in_dim, out_dim, bias)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x, adj):
        for layer in self.gcn_layers[:-1]:
            x = nn.relu(layer(x, adj))
            x = self.dropout(x)

        x = self.gcn_layers[-1](x, adj)
        return x