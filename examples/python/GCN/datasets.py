import os
import requests
import tarfile

import numpy as np
import scipy.sparse as sparse

"""
Preprocessing follows the same implementation as in:
https://github.com/tkipf/gcn
https://github.com/senadkurtisi/pytorch-GCN/tree/main
"""


def download_cora():
    """Downloads the cora dataset into a local cora folder."""

    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    extract_to = "."

    if os.path.exists(os.path.join(extract_to, "cora")):
        return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(extract_to, url.split("/")[-1])

        # Write the file to local disk
        with open(file_path, "wb") as file:
            file.write(response.raw.read())

        # Extract the .tgz file
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"Cora dataset extracted to {extract_to}")

        os.remove(file_path)


def train_val_test_mask(labels, num_classes):
    """Splits the loaded dataset into train/validation/test sets."""

    train_set = list(range(140))
    validation_set = list(range(200, 500))
    test_set = list(range(500, 1500))

    return train_set, validation_set, test_set


def enumerate_labels(labels):
    """Converts the labels from the original
    string form to the integer [0:MaxLabels-1]
    """
    unique = list(set(labels))
    labels = np.array([unique.index(label) for label in labels])
    return labels


def normalize_adjacency(adj):
    """Normalizes the adjacency matrix according to the
    paper by Kipf et al.
    https://arxiv.org/pdf/1609.02907.pdf
    """
    adj = adj + sparse.eye(adj.shape[0])

    node_degrees = np.array(adj.sum(1))
    node_degrees = np.power(node_degrees, -0.5).flatten()
    node_degrees[np.isinf(node_degrees)] = 0.0
    node_degrees[np.isnan(node_degrees)] = 0.0
    degree_matrix = sparse.diags(node_degrees, dtype=np.float32)

    adj = degree_matrix @ adj @ degree_matrix
    return adj


def load_data(config):
    """Loads the Cora graph data into MLX array format."""
    print("Loading Cora dataset...")

    # Graph nodes
    raw_nodes_data = np.genfromtxt(config.nodes_path, dtype="str")
    raw_node_ids = raw_nodes_data[:, 0].astype(
        "int32"
    )  # unique identifier of each node
    raw_node_labels = raw_nodes_data[:, -1]
    labels_enumerated = enumerate_labels(raw_node_labels)  # target labels as integers
    node_features = sparse.csr_matrix(raw_nodes_data[:, 1:-1], dtype="float32")

    # Edges
    ids_ordered = {raw_id: order for order, raw_id in enumerate(raw_node_ids)}
    raw_edges_data = np.genfromtxt(config.edges_path, dtype="int32")
    edges_ordered = np.array(
        list(map(ids_ordered.get, raw_edges_data.flatten())), dtype="int32"
    ).reshape(raw_edges_data.shape)

    # Adjacency matrix
    adj = sparse.coo_matrix(
        (np.ones(edges_ordered.shape[0]), (edges_ordered[:, 0], edges_ordered[:, 1])),
        shape=(labels_enumerated.shape[0], labels_enumerated.shape[0]),
        dtype=np.float32,
    )

    adj = adj + adj.T.multiply(adj.T > adj)
    adj = normalize_adjacency(adj)

    adj_coo = adj.tocoo()

    # Convert the COO matrix to an edge index
    edge_index = np.vstack((adj_coo.row, adj_coo.col))

    # Make the adjacency matrix symmetric
    # edge_index = np.stack([edges_ordered[:, 0], edges_ordered[:, 1]])

    # reverse_edges = edge_index[::-1, :]
    # symmetric_edge_index = np.concatenate([edge_index, reverse_edges], axis=1)
    # symmetric_edge_index = np.unique(symmetric_edge_index, axis=1)


    # edge_index = symmetric_edge_index

    print("Dataset loaded.")
    return node_features.toarray(), labels_enumerated, edge_index