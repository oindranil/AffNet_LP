import os
import pandas as pd
import numpy as np
import re
from .base_dataset import BaseDataset

class MovieLens1M(BaseDataset):
    """
    MovieLens 1M dataset loader. Inherits from BaseDataset.
    Supports negative sampling via `neg_ratio`.
    """

    def __init__(self, data_dir, neg_ratio=None):
        super().__init__()
        self.neg_ratio = neg_ratio
        self.load(os.path.join(data_dir, 'ml-1m'))

    def load(self, data_dir):
        # Load ratings.dat
        ratings_path = os.path.join(data_dir, 'ratings.dat')
        ratings = pd.read_csv(ratings_path, sep='::', engine='python',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'])

        # Load movies.dat
        movies_path = os.path.join(data_dir, 'movies.dat')
        movies = pd.read_csv(movies_path, sep='::', engine='python',
                             names=['movie_id', 'title', 'genres'],
                             encoding='latin-1')

        # Extract release year
        movies['year'] = movies['title'].apply(
            lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else 0)

        # Genre one-hot
        all_genres = sorted({g for genre_str in movies['genres'] for g in genre_str.split('|')})
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}

        def encode_genres(genres_str):
            vec = np.zeros(len(all_genres), dtype=np.float32)
            for g in genres_str.split('|'):
                vec[genre_to_idx[g]] = 1.0
            return vec

        genre_features = np.stack(movies['genres'].apply(encode_genres))

        # Year one-hot
        min_year, max_year = 1910, 2025
        year_bins = list(range(min_year, max_year + 10, 10))
        year_indices = np.digitize(movies['year'], year_bins)
        year_one_hot = np.zeros((len(movies), len(year_bins)), dtype=np.float32)
        year_one_hot[np.arange(len(movies)), year_indices - 1] = 1.0

        item_features = np.hstack([genre_features, year_one_hot])

        # Remap user/item IDs to 0-based indices
        unique_users = ratings['user_id'].unique()
        unique_items = ratings['movie_id'].unique()
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        ratings['user_idx'] = ratings['user_id'].map(self.user_id_map)
        ratings['item_idx'] = ratings['movie_id'].map(self.item_id_map)

        # Binarize ratings
        ratings['label'] = (ratings['rating'] >= 4).astype(np.float32)

        # Positive edges
        pos_edges = ratings[['user_idx', 'item_idx']].to_numpy()
        pos_labels = ratings['label'].values.astype(np.float32)
        num_pos = len(pos_edges)

        # Sample negative edges
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)

        if self.neg_ratio is not None:
            num_neg = int(num_pos * self.neg_ratio)
            user_item_set = set(map(tuple, pos_edges))  # For checking duplicates
            neg_edges = set()
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility

            while len(neg_edges) < num_neg:
                u = rng.integers(0, num_users)
                i = rng.integers(0, num_items)
                if (u, i) not in user_item_set:
                    neg_edges.add((u, i))

            neg_edges = np.array(list(neg_edges), dtype=np.int64)
            neg_labels = np.zeros(len(neg_edges), dtype=np.float32)

            # Combine
            all_edges = np.vstack([pos_edges, neg_edges])
            all_weights = np.concatenate([pos_labels, neg_labels])

        else:
            all_edges = pos_edges
            all_weights = pos_labels
            
        # Reorder item features to match internal indexing
        movie_id_to_row_idx = {mid: i for i, mid in enumerate(movies['movie_id'])}
        reorder_indices = [movie_id_to_row_idx[mid] for mid in unique_items]
        item_features = item_features[reorder_indices]

        # Assign attributes
        self.dataset_name = 'ML-1M'
        self.user_item_edges = all_edges.T  # shape: [2, num_edges]
        self.edge_weights = all_weights     # shape: [num_edges]
        self.item_features = item_features
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = item_features.shape[1]
        self.num_edges = all_edges.shape[0]
        self.has_features = True