# datasets/base_dataset.py

class BaseDataset:
    """
    Abstract base class for graph-based recommender datasets.
    All dataset loaders must inherit from this class.
    """

    def __init__(self):
        self.dataset_name = None

        # Core graph elements
        self.user_item_edges = None     # shape: [2, num_edges]
        self.edge_weights = None        # shape: [num_edges]
        self.item_features = None       # shape: [num_items, num_features]

        # Metadata
        self.num_users = 0
        self.num_items = 0
        self.num_features = 0
        self.num_edges = 0

        # Optional ID mappings (internal use)
        self.user_id_map = {}           # original_user_id → index
        self.item_id_map = {}           # original_item_id → index

    def summary(self):
        print(f"# Users     : {self.num_users}")
        print(f"# Items     : {self.num_items}")
        print(f"# Features  : {self.num_features}")
        print(f"# Edges     : {self.num_edges}")
        print(f"Edge Index  : {self.user_item_edges.shape}")
        print(f"Item Feats  : {self.item_features.shape}")
        print("===========================")

    def get_edge_index(self):
        """Returns the [2, num_edges] edge index (user -> item)."""
        return self.user_item_edges

    def get_edge_weights(self):
        """Returns edge weights (e.g., ratings)."""
        return self.edge_weights

    def get_item_features(self):
        """Returns [num_items, num_features] array."""
        return self.item_features

    def get_metadata(self):
        """Returns num_users, num_items, num_features as a tuple."""
        return self.num_users, self.num_items, self.num_features
