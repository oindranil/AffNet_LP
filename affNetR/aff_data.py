import tensorflow as tf
import numpy as np

class AffData:
    def __init__(self, dataset):
        """
        Construct AffData from a dataset (e.g., MovieLens1M object).

        Args:
            dataset: An instance of a dataset inheriting from BaseDataset
        """
        self._build(dataset)

    def _build(self, dataset):
        # Unpack raw data
        user_item_edges = dataset.user_item_edges      # shape: [2, num_edges]
        edge_weights = dataset.edge_weights            # shape: [num_edges]
        item_features = dataset.item_features          # shape: [num_items, num_features]

        # Validate binary weights
        assert set(np.unique(edge_weights)).issubset({0.0, 1.0}), "Edge weights must be binary (0 or 1)."

        # TensorFlow sparse indices (convert shape: [2, num_edges] → [num_edges, 2])
        indices = np.stack([user_item_edges[0], user_item_edges[1]], axis=1).astype(np.int64)
        values = edge_weights.astype(np.float32)
        dense_shape = [dataset.num_users, dataset.num_items]

        # SparseTensor for edge_index
        edge_index = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        edge_index = tf.sparse.reorder(edge_index)

        # Item features tensor
        if item_features is None:
            x = None
            num_features = 0
        else:
            x = tf.convert_to_tensor(item_features, dtype=tf.float32)
            num_features = x.shape[1]

        # Store members
        self.dataset_name = dataset.dataset_name
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_features = num_features
        self.num_edges = user_item_edges.shape[1]
        self.x = x
        self.edge_index = edge_index

    def split(self, test_frac=0.2, seed: int = 42):
        edge_index = self.edge_index
        indices = edge_index.indices
        values = edge_index.values

        pos_mask = tf.where(values == 1.0)[:, 0]
        neg_mask = tf.where(values == 0.0)[:, 0]

        rng = np.random.default_rng(seed)

        def stratified_split(mask):
            mask_np = mask.numpy()
            perm = rng.permutation(len(mask_np))
            split_point = int(len(mask_np) * (1 - test_frac))
            return mask_np[perm[:split_point]], mask_np[perm[split_point:]]

        pos_train, pos_test = stratified_split(pos_mask)
        neg_train, neg_test = stratified_split(neg_mask)

        train_idx = np.concatenate([pos_train, neg_train])
        test_idx = np.concatenate([pos_test, neg_test])

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        def build_sparse(idx_subset):
            selected_indices = tf.gather(indices, idx_subset)
            selected_values = tf.gather(values, idx_subset)
            return tf.sparse.reorder(tf.sparse.SparseTensor(
                indices=selected_indices,
                values=selected_values,
                dense_shape=edge_index.dense_shape
            ))

        train_edge = build_sparse(train_idx)
        test_edge = build_sparse(test_idx)

        # Edge counts
        num_train_edges = train_edge.indices.shape[0]
        num_test_edges = test_edge.indices.shape[0]

        # Create AffData instances
        train_data = AffData.__new__(AffData)
        train_data.dataset_name = self.dataset_name
        train_data.num_users = self.num_users
        train_data.num_items = self.num_items
        train_data.num_features = self.num_features
        train_data.num_edges = num_train_edges
        train_data.x = self.x
        train_data.edge_index = train_edge

        test_data = AffData.__new__(AffData)
        test_data.dataset_name = self.dataset_name
        test_data.num_users = self.num_users
        test_data.num_items = self.num_items
        test_data.num_features = self.num_features
        test_data.num_edges = num_test_edges
        test_data.x = self.x
        test_data.edge_index = test_edge

        return train_data, test_data

    def count_isolated_nodes(self):
        indices = self.edge_index.indices.numpy()
        values = self.edge_index.values.numpy()
        
        pos_indices = indices[values > 0]  # only positive edges
    
        users_with_edges = np.unique(pos_indices[:, 0])
        items_with_edges = np.unique(pos_indices[:, 1])
    
        num_isolated_users = self.num_users - len(users_with_edges)
        num_isolated_items = self.num_items - len(items_with_edges)
    
        print(f"Isolated users: {num_isolated_users}/{self.num_users}")
        print(f"Isolated items: {num_isolated_items}/{self.num_items}")
    
    def user_item_edge_counts(self):
        indices = self.edge_index.indices.numpy()
        values = self.edge_index.values.numpy()
    
        # Only consider positive edges
        pos_indices = indices[values > 0]
    
        # Count edges per user and per item
        user_counts = np.bincount(pos_indices[:, 0], minlength=self.num_users)
        item_counts = np.bincount(pos_indices[:, 1], minlength=self.num_items)
    
        print(f"User degrees: min={user_counts.min()}, max={user_counts.max()}, avg={user_counts.mean():.2f}")
        print(f"Item degrees: min={item_counts.min()}, max={item_counts.max()}, avg={item_counts.mean():.2f}")
    
    def summary(self):
        print(f"\nSummary: {self.dataset_name}")
        print(f" - Users      : {self.num_users}")
        print(f" - Items      : {self.num_items}")
        print(f" - Features   : {self.num_features}")
        print(f" - Edges      : {self.num_edges}")
        print(f" - edge_index : {self.edge_index.dense_shape}")
        print()

    def induced_subgraph_on_top_items(self, top_k_items=5000, max_users=5000):
        """
        Induce a subgraph from the top-K most popular items and the top-M most active users
        (based on positive interactions), including both positive and negative edges.
    
        Args:
            top_k_items (int): Number of most popular items to keep.
            max_users (int): Max number of most active users to include.
        """
        indices = self.edge_index.indices.numpy()   # [num_edges, 2]
        values = self.edge_index.values.numpy()     # [num_edges]
    
        # Step 1: Filter positive edges
        pos_edges = indices[values == 1.0]
    
        # Step 2: Count item popularity and get top-K items
        item_degrees = np.bincount(pos_edges[:, 1], minlength=self.num_items)
        top_item_indices = np.argsort(-item_degrees)[:top_k_items]
        top_item_set = set(top_item_indices)
    
        # Step 3: Filter positive edges to only include top items
        item_mask = np.isin(pos_edges[:, 1], top_item_indices)
        filtered_pos_edges = pos_edges[item_mask]
    
        # Step 4: Count user activity and get top-M users
        user_degrees = np.bincount(filtered_pos_edges[:, 0], minlength=self.num_users)
        active_user_indices = np.argsort(-user_degrees)
        active_user_indices = active_user_indices[user_degrees[active_user_indices] > 0]
        top_user_indices = active_user_indices[:max_users]
        top_user_set = set(top_user_indices)
    
        # Step 5: Apply joint filter (on all edges — pos & neg) using top user/item sets
        all_edges = indices
        all_values = values
        mask = np.array([
            (u in top_user_set) and (i in top_item_set)
            for u, i in all_edges
        ])
        final_edges = all_edges[mask]
        final_values = all_values[mask]
    
        # Step 6: Remap user/item IDs to contiguous range
        user_id_map = {old: new for new, old in enumerate(sorted(top_user_set))}
        item_id_map = {old: new for new, old in enumerate(sorted(top_item_set))}
    
        remapped_edges = np.array([
            [user_id_map[u], item_id_map[i]]
            for u, i in final_edges
        ], dtype=np.int64)
    
        # Step 7: Create SparseTensor for the new subgraph
        new_sparse_edge_index = tf.sparse.SparseTensor(
            indices=remapped_edges,
            values=final_values.astype(np.float32),
            dense_shape=[len(user_id_map), len(item_id_map)]
        )
        new_sparse_edge_index = tf.sparse.reorder(new_sparse_edge_index)
    
        # Step 8: Filter item features if available
        if self.x is not None:
            sorted_item_indices = sorted(top_item_set)
            new_item_features = tf.gather(self.x, sorted_item_indices)
        else:
            new_item_features = None
    
        # Step 9: Create the new AffData object
        subgraph = AffData.__new__(AffData)
        subgraph.dataset_name = self.dataset_name + "_subgraph"
        subgraph.num_users = len(user_id_map)
        subgraph.num_items = len(item_id_map)
        subgraph.num_features = self.num_features
        subgraph.num_edges = remapped_edges.shape[0]
        subgraph.edge_index = new_sparse_edge_index
        subgraph.x = new_item_features
    
        return subgraph

