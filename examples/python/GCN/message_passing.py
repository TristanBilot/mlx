import mlx.core as mx
import mlx.nn as nn


class MessagePassing(nn.Module):
    def __init__(self, aggr=None):
        super().__init__()

        self.aggr = aggr

    def __call__(self, x, edge_index, **kwargs):
        pass

    def propagate(self, x, edge_index, **kwargs):
        # process arguments and create *_kwargs

        src_idx, dst_idx = edge_index
        x_i = x[src_idx]
        x_j = x[dst_idx]

        # Message
        messages = self.message(x_i, x_j)  # **msg_kwargs)

        # Aggregate
        aggregated = self.aggregate(messages, dst_idx)  # **agg_kwargs)

        # Update
        output = self.update(aggregated)  # **upd_kwargs)

        return output

    def message(self, x_i, x_j, **kwargs):
        return x_i

    def aggregate(self, messages, indices, **kwargs):
        if self.aggr == "add":
            nb_unique_indices = _unique(indices)
            empty_tensor = mx.zeros((nb_unique_indices, messages.shape[-1]))
            update_dim = (messages.shape[0], 1, messages.shape[1])
            return mx.scatter_add(
                empty_tensor, indices, messages.reshape(update_dim), 0
            )

    def update(self, aggregated, **kwargs):
        return aggregated


def _unique(array):
    return len(set(array.tolist()))
