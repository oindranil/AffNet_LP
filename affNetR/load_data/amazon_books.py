# datasets/amazon_book.py

import os
import numpy as np
from .base_dataset import BaseDataset
from torch_geometric.datasets import AmazonBook
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import ToUndirected

class Amazon_Book(BaseDataset):
    """
    Loads the AmazonBook dataset via PyG and adapts it to BaseDataset format,
    including negative sampling at a flexible ratio.
    """
    def __init__(self, data_dir, neg_ratio=1.0):
        super().__init__()
        self.neg_ratio = neg_ratio
        self.load(os.path.join(data_dir, "amazon_books"))

    def load(self, root):
        # Load dataset and ensure symmetry
        pyg_data = AmazonBook(root=root, transform=ToUndirected())[0]

        pos_edge_index = pyg_data[('user', 'rates', 'book')].edge_index
        raw_u = pos_edge_index[0].numpy()
        raw_i = pos_edge_index[1].numpy()

        # Map to contiguous indices
        unique_u = np.unique(raw_u)
        unique_i = np.unique(raw_i)
        num_users = len(unique_u)
        num_items = len(unique_i)
        u_map = {u: idx for idx, u in enumerate(unique_u)}
        i_map = {i: idx for idx, i in enumerate(unique_i)}

        users = np.vectorize(u_map.get)(raw_u)
        items = np.vectorize(i_map.get)(raw_i)
        pos_edges = np.stack([users, items], axis=1)

        num_pos = pos_edges.shape[0]
        num_neg = int(num_pos * self.neg_ratio)

        # Sample negative edges using PyG's utility
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(num_users, num_items),
            num_neg_samples=num_neg,
            method='sparse'
        )
        neg_u = np.array([u_map[u] for u in neg_edge_index[0].numpy()], dtype=np.int64)
        neg_i = np.array([i_map[i] for i in neg_edge_index[1].numpy()], dtype=np.int64)
        neg_edges = np.stack([neg_u, neg_i], axis=1)

        # Build combined edges and weights
        all_edges = np.vstack([pos_edges, neg_edges])
        all_weights = np.concatenate([
            np.ones(num_pos, dtype=np.float32),
            np.zeros(neg_edges.shape[0], dtype=np.float32)
        ])

        # Dummy item features (none available)
        item_features = None
        num_features = 0
        
        # Assign BaseDataset fields
        self.dataset_name = 'AmazonBook'
        self.user_id_map = u_map
        self.item_id_map = i_map
        self.user_item_edges = all_edges.T
        self.edge_weights = all_weights
        self.item_features = item_features
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.num_edges = all_edges.shape[0]
        self.has_features = False