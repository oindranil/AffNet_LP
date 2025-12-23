"""
make data sparse - each feature must have at least p fraction of rows set to zero
inputs: PyG data object, fraction p
"""
import torch

def sparsify_features(x, sparsity):
    """
    Sparsify the node features in a PyG graph by randomly setting p% of entries 
    in each column (feature) to zero.

    Args:
        data (Data): PyG graph data object with `x` (node features).
        sparsity (float): Fraction of entries to zero out (0 < sparsity < 1).
        seed (int): Random seed for reproducibility.

    Returns:
        Data: New Data object with sparsified `x`.
    """
    
    num_nodes, num_features = x.shape

    # Create mask with 1s everywhere initially
    mask = torch.ones_like(x, dtype=torch.bool)

    # For each feature (column), randomly zero out p% of rows
    for j in range(num_features):
        # number of nodes to drop in this column
        num_to_zero = int(sparsity * num_nodes)
        
        if num_to_zero > 0:
            idx = torch.randperm(num_nodes)[:num_to_zero]
            mask[idx, j] = 0

    # Apply mask
    x = x * mask.float()
    
    return x


# Example usage:
# Assume you have a dataset like Cora, PubMed, etc.
# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0]

# Sparsify 10% per feature column
# data_sparse = sparsify_features(data, p=0.1)
