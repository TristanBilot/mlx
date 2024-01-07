from argparse import ArgumentParser
from time import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from datasets import download_cora, load_data, train_val_test_mask
from gcn import GCN


def loss_fn(y_hat, y, weight_decay=0.0, parameters=None):
    l = mx.mean(nn.losses.cross_entropy(y_hat, y))

    if weight_decay != 0.0:
        assert parameters != None, "Model parameters missing for L2 reg."

        l2_reg = sum(mx.sum(p[1] ** 2) for p in tree_flatten(parameters)).sqrt()
        return l + weight_decay * l2_reg

    return l


def eval_fn(x, y):
    return mx.mean(mx.argmax(x, axis=1) == y)


def forward_fn(gcn, x, adj, y, train_mask, weight_decay):
    y_hat = gcn(x, adj)
    loss = loss_fn(y_hat[train_mask], y[train_mask], weight_decay, gcn.parameters())
    return loss, y_hat


def to_mlx(x, y, adj, train_mask, val_mask, test_mask):
    x = mx.array(x.tolist(), mx.float32)
    y = mx.array(y.tolist(), mx.int32)
    adj = mx.array(adj.tolist())
    train_mask = mx.array(train_mask.tolist())
    val_mask = mx.array(val_mask.tolist())
    test_mask = mx.array(test_mask.tolist())
    return x, y, adj, train_mask, val_mask, test_mask


def get_masks(train_mask, val_mask, test_mask):
    train_mask = mx.array([i for i, e in enumerate(train_mask) if e == True])
    val_mask = mx.array([i for i, e in enumerate(val_mask) if e == True])
    test_mask = mx.array([i for i, e in enumerate(test_mask) if e == True])

    return (train_mask, val_mask, test_mask)


def main(args):
    # Data loading
    dataset = Planetoid(root="tmp/Cora", name="Cora")
    data = dataset[0]

    x, y, adj = data.x, data.y, data.edge_index
    train_mask, val_mask, test_mask = get_masks(
        data.train_mask, data.val_mask, data.test_mask
    )

    x, y, adj, train_mask, val_mask, test_mask = to_mlx(
        x, y, adj, train_mask, val_mask, test_mask
    )

    gcn = GCN(
        x_dim=x.shape[-1],
        h_dim=args.hidden_dim,
        out_dim=args.nb_classes,
        nb_layers=args.nb_layers,
        dropout=args.dropout,
        bias=args.bias,
    )
    mx.eval(gcn.parameters())

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad_fn = nn.value_and_grad(gcn, forward_fn)

    best_val_loss = float("inf")
    cnt = 0
    times = []

    # Training loop
    for epoch in range(args.epochs):
        start = time()

        # Loss
        (loss, y_hat), grads = loss_and_grad_fn(
            gcn, x, adj, y, train_mask, args.weight_decay
        )
        optimizer.update(gcn, grads)
        mx.eval(gcn.parameters(), optimizer.state)

        # Validation
        val_loss = loss_fn(y_hat[val_mask], y[val_mask])
        val_acc = eval_fn(y_hat[val_mask], y[val_mask])

        times.append(time() - start)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cnt = 0
        else:
            cnt += 1
            if cnt == args.patience:
                break

        print(
            " | ".join(
                [
                    f"Epoch: {epoch:3d}",
                    f"Train loss: {loss.item():.3f}",
                    f"Val loss: {val_loss.item():.3f}",
                    f"Val acc: {val_acc.item():.2f}",
                ]
            )
        )

    # Test
    test_y_hat = gcn(x, adj)
    test_loss = loss_fn(y_hat[test_mask], y[test_mask])
    test_acc = eval_fn(y_hat[test_mask], y[test_mask])
    mean_time = sum(times) / len(times)

    print(f"Test loss: {test_loss.item():.3f}  |  Test acc: {test_acc.item():.2f}")
    print(f"Mean time: {mean_time:.5f}")
    return mean_time


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nodes_path", type=str, default="cora/cora.content")
    parser.add_argument("--edges_path", type=str, default="cora/cora.cites")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--nb_layers", type=int, default=2)
    parser.add_argument("--nb_classes", type=int, default=7)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    main(args)
