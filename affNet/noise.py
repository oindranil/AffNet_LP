"""
program: Feature Noise Injection Utility
description:
    This script adds controlled noise to node feature tensors for robustness testing in GNNs.
    It classifies features into continuous, binary, and categorical types based on their values.
    Noise is added as follows:
      - Continuous: Gaussian noise
      - Binary: 0/1 flip
      - Categorical: random dropout   
"""

import torch

# ---------------------------
# Step 1: Feature Classifier
# ---------------------------
def classify_feature_column(col, float_tol=1e-3, cat_thresh=50):
    unique_vals = torch.unique(col)
    num_unique = unique_vals.numel()
    
    # Binary check
    if num_unique <= 2:
        if all(torch.isclose(unique_vals, torch.tensor([0.0]), atol=float_tol) |
               torch.isclose(unique_vals, torch.tensor([1.0]), atol=float_tol)):
            return "binary"
    
    # Categorical check
    if num_unique <= cat_thresh:
        frac_int_like = torch.mean(((unique_vals - unique_vals.round()).abs() < float_tol).float())
        if frac_int_like > 0.9:
            return "categorical"
    
    return "continuous"


def classify_all_features(x, float_tol=1e-3, cat_thresh=50):
    feature_types = []
    for i in range(x.size(1)):
        ftype = classify_feature_column(x[:, i], float_tol, cat_thresh)
        feature_types.append(ftype)
    return feature_types  # list of strings of length x.size(1)

# ---------------------------
# Step 2: Noise Routines
# ---------------------------

def add_gaussian_noise(x, cols, mean=0.0, std=0.1):
    noise = torch.randn_like(x[:, cols]) * std + mean
    x[:, cols] += noise
    return x

def flip_binary_features(x, cols):
    x[:, cols] = 1.0 - x[:, cols]
    return x

def apply_feature_dropout(x, drop_prob=0.1):
    mask = torch.rand_like(x) > drop_prob
    return x * mask

# ---------------------------
# Step 3: Master Noise Function
# ---------------------------

def apply_feature_noise(x, gaussian_std=0.1, dropout_prob=0.1):
    """
    Adds Gaussian noise to continuous features, flips binary ones, and drops values randomly in all features.
    Returns:
        - noisy feature tensor
        - fraction of features (columns) that had Gaussian or Flip noise applied
        - feature type list
    """
    x = x.clone()
    feature_types = classify_all_features(x)
    
    # Get indices by type
    continuous_cols = [i for i, t in enumerate(feature_types) if t == 'continuous']
    binary_cols     = [i for i, t in enumerate(feature_types) if t == 'binary']
    # categorical_cols = [i for i, t in enumerate(feature_types) if t == 'categorical']  # skipped

    # Apply targeted noise
    if continuous_cols:
        x = add_gaussian_noise(x, continuous_cols, std=gaussian_std)
    if binary_cols:
        x = flip_binary_features(x, binary_cols)

    # Apply universal dropout
    x = apply_feature_dropout(x, drop_prob=dropout_prob)

    return x

