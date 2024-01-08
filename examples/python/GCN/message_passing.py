import mlx.core as mx
import mlx.nn as nn


def degree(index, num_edges):
    out = mx.zeros((num_edges,))
    one = mx.ones((index.shape[0],), dtype=out.dtype)
    return mx.scatter_add(out, index, one.reshape(-1, 1), 0)


class MessagePassing(nn.Module):
    def __init__(self, aggr=None):
        super().__init__()

        self.aggr = aggr

    def __call__(self, x, edge_index, **kwargs):
        raise NotImplementedError

    def propagate(self, x, edge_index, **kwargs):
        # process arguments and create *_kwargs

        src_idx, dst_idx = edge_index
        x_i = x[src_idx]
        x_j = x[dst_idx]

        row, col = edge_index
        deg = degree(col, x.shape[0])
        deg_inv_sqrt = deg ** (-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message
        messages = self.message(x_i, x_j, norm)  # **msg_kwargs)

        # Aggregate
        aggregated = self.aggregate(messages, dst_idx)  # **agg_kwargs)

        # Update
        output = self.update_(aggregated)  # **upd_kwargs)

        return output

    def message(self, x_i, x_j, norm, **kwargs):
        return norm.reshape(-1, 1) * x_j

    def aggregate(self, messages, indices, **kwargs):
        if self.aggr == "add":
            nb_unique_indices = _unique(indices)
            empty_tensor = mx.zeros((nb_unique_indices, messages.shape[-1]))
            update_dim = (messages.shape[0], 1, messages.shape[1])
            return mx.scatter_add(
                empty_tensor, indices, messages.reshape(update_dim), 0
            )

    def update_(self, aggregated, **kwargs):
        return aggregated


def _unique(array):
    return len(set(array.tolist()))
