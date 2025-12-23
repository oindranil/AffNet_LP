import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


def estimate_num_hops_from_roots(
    num_nodes: int,
    num_edges: int,
    max_nodes: int,
    num_roots: int,
    fanout_ratio: float = 0.5
) -> tuple[int, int]:
    """
    Estimate number of hops needed to reach ~max_nodes nodes starting from num_roots,
    using average graph degree and fanout_ratio.
    """
    avg_deg = (2 * num_edges) / num_nodes  # For undirected graphs
    fanout = max(1, int(avg_deg * fanout_ratio))

    r = num_roots
    f = fanout

    if f == 1:
        max_h = (max_nodes // r) - 1
    else:
        max_h = 0
        while True:
            nodes_est = r * ((f ** (max_h + 1) - 1) // (f - 1))
            if nodes_est > max_nodes:
                break
            max_h += 1

    return max_h, fanout


def truncate_subgraph(batch: Data, max_nodes: int) -> Data:
    """Truncates a sampled subgraph to max_nodes nodes, remapping edge indices."""
    keep_idx = torch.arange(max_nodes)
    node_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
    node_mask[keep_idx] = True

    old_to_new = -torch.ones(batch.num_nodes, dtype=torch.long)
    old_to_new[keep_idx] = torch.arange(max_nodes)

    edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
    edge_index = batch.edge_index[:, edge_mask]
    edge_index = old_to_new[edge_index]

    # Safe handling for optional `y`
    y = batch.y[keep_idx] if hasattr(batch, 'y') and batch.y is not None else None

    return Data(
        x=batch.x[keep_idx],
        y=y,
        edge_index=edge_index,
        num_nodes=max_nodes
    )


def create_subgraphs(
    data: Data,
    max_parts: int,
    max_nodes: int,
    num_roots: int,
    fanout_ratio: float = 0.5
) -> list[Data]:
    """
    Sample up to `max_parts` subgraphs from `data`, each with ≤ max_nodes nodes,
    starting from `num_roots` per batch and adapting hops/fanout accordingly.
    """
    num_hops, fanout = estimate_num_hops_from_roots(
        num_nodes=data.num_nodes,
        num_edges=data.num_edges,
        max_nodes=max_nodes * 2, # estimate for double max nodes - we are anyway truncating if more
        num_roots=num_roots,
        fanout_ratio=fanout_ratio
    )
    num_neighbors = [fanout] * num_hops

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=num_roots,
        shuffle=True,
        input_nodes=None
    )

    subgraphs = []
    for batch in loader:
        if batch.num_nodes > max_nodes:
            subgraph = truncate_subgraph(batch, max_nodes)
        else:
            subgraph = Data(
                x=batch.x,
                y=batch.y,
                edge_index=batch.edge_index
            )
        subgraphs.append(subgraph)
        if len(subgraphs) >= max_parts:
            break

    return subgraphs
