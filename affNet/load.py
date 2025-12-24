"""
program: load, subgraph and split routines

"""

# import libraries
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, HeterophilousGraphDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
import tensorflow as tf
import torch
import numpy as np
import random

def convert_edge_index_to_tf(edge_index, n_nodes, n_edges):
    edge_index = tf.cast(edge_index, tf.int64)
    edge_index = tf.sparse.SparseTensor(indices=tf.transpose(edge_index), 
                     values=tf.ones(n_edges), dense_shape=[n_nodes, n_nodes])
    edge_index = tf.sparse.reorder(edge_index)
    return(edge_index)

class tf_GData:
    def __init__(self, torch_GData):

        # ------------------------
        # Node features / labels
        # ------------------------
        self.x = tf.cast(torch_GData.x, tf.float32)

        if hasattr(torch_GData, 'y') and torch_GData.y is not None:
            self.y = tf.cast(torch_GData.y, tf.float32)
        else:
            self.y = None

        # ------------------------
        # TRAINING EDGES ONLY
        # ------------------------
        if hasattr(torch_GData, 'edge_index'):
            self.edge_index = convert_edge_index_to_tf(
                torch_GData.edge_index,
                torch_GData.num_nodes,
                torch_GData.num_edges
            )

        if hasattr(torch_GData, 'pos_edge_index'):
            self.pos_edge_index = convert_edge_index_to_tf(
                torch_GData.pos_edge_index,
                torch_GData.num_nodes,
                torch_GData.n_pos
            )

        if hasattr(torch_GData, 'neg_edge_index'):
            neg = torch_GData.neg_edge_index
            if neg.dim() == 2 and neg.shape[0] == 2:
                self.neg_edge_index = convert_edge_index_to_tf(
                    neg,
                    torch_GData.num_nodes,
                    torch_GData.n_neg
                )

        # ------------------------
        # METADATA
        # ------------------------
        self.num_nodes = torch_GData.num_nodes
        self.num_features = torch_GData.num_features
        self.num_edges = torch_GData.num_edges
        self.n_pos = torch_GData.n_pos
        self.n_neg = torch_GData.n_neg

        # ------------------------
        # OGB EVALUATION PAYLOAD
        # (DO NOT TOUCH / DO NOT TF-CONVERT)
        # ------------------------
        for key in [
            "ogb_test_pos_edge_index",
            "ogb_test_neg_edge_index",
        ]:
            if hasattr(torch_GData, key):
                setattr(self, key, getattr(torch_GData, key))

# convert a graph dataset from torch to tf
def convert_data_to_tf(data):
    def convert_edge_index_to_tf(edge_index, n_nodes, n_edges):
        edge_index = tf.cast(edge_index, tf.int64)
        edge_index = tf.sparse.SparseTensor(indices=tf.transpose(edge_index), 
                         values=tf.ones(n_edges), dense_shape=[n_nodes, n_nodes])
        edge_index = tf.sparse.reorder(edge_index)
        return(edge_index)
    
    data.x = tf.cast(data.x, tf.float32)
    if hasattr(data, 'y'):
        data.y = tf.cast(data.y, tf.float32)
    if data.edge_index is not None:
        data.edge_index = convert_edge_index_to_tf(data.edge_index, data.num_nodes, data.num_edges)
    if hasattr(data, 'pos_edge_index'):
        data.pos_edge_index = convert_edge_index_to_tf(data.pos_edge_index, data.num_nodes, data.n_pos)
    if hasattr(data, 'neg_edge_index'):
        data.neg_edge_index = convert_edge_index_to_tf(data.neg_edge_index, data.num_nodes, data.n_neg)
    data.num_features = data.x.shape[1]
    return(data)


# create trtain and test masks where not present
def get_mask(n_nodes):
    mask = np.array([1]*n_nodes, dtype=bool)
    step =  10 # 10% for test, uniform & not random to ensure reproducability 
    for i in range(round(n_nodes/step)):
        mask[step*i] = False
    train_mask = tf.reshape(tf.convert_to_tensor(mask), (-1,1))
    test_mask = tf.math.logical_not(train_mask)
    return(train_mask, test_mask)

def normalize_adj(edge_index, n_nodes):
    def d_half_inv(e, n_nodes):
        deg = [e.count(i) for i in range(n_nodes)]
        deg_inv_sqrt = [1/np.sqrt(d) if d!=0 else 0 for d in deg]
        return(deg_inv_sqrt)
    edges = edge_index.indices.numpy()
    vals = list(edge_index.values.numpy())
    n_edges = len(edges)
    deg_inv_sqrt_0 = d_half_inv(list(edges[:,0]), n_nodes)
    deg_inv_sqrt_1 = d_half_inv(list(edges[:,1]), n_nodes)
    v_norm = [vals[i]*deg_inv_sqrt_0[edges[i,0]]*deg_inv_sqrt_1[edges[i,1]] 
              for i in range(n_edges)]
    edge_index = tf.sparse.SparseTensor(indices=edge_index.indices, 
                    values=v_norm, dense_shape=[n_nodes, n_nodes])
    return(edge_index)

def sample_negatives_degree_distance(
    pos_edge_index,
    num_nodes,
    adj_list,
    degrees,
    K,):

    device = pos_edge_index.device
    E = pos_edge_index.size(1)

    neg_src = []
    neg_dst = []

    for i in range(E):
        u, v = pos_edge_index[:, i].tolist()

        # choose anchor with higher degree
        anchor = u if degrees[u] >= degrees[v] else v

        candidates = []
        attempts = 0

        while len(candidates) < K and attempts < K * 20:
            w = torch.randint(0, num_nodes, (1,), device=device).item()
            attempts += 1

            if w == anchor:
                continue
            if w in adj_list[anchor]:
                continue

            candidates.append(w)

        # fallback (rare)
        if len(candidates) < K:
            for w in range(num_nodes):
                if w != anchor and w not in adj_list[anchor]:
                    candidates.append(w)
                if len(candidates) == K:
                    break

        for w in candidates:
            neg_src.append(anchor)
            neg_dst.append(w)

    return torch.tensor([neg_src, neg_dst], dtype=torch.long, device=device)

def generate_neg_edges_per_pos(pos_edge_index, num_nodes, forbidden_edge_index, K):
    device = pos_edge_index.device
    E = pos_edge_index.shape[1]

    forbidden = set(
        (u.item(), v.item()) for u, v in forbidden_edge_index.t()
    )

    neg_edges = torch.empty((E, K, 2), dtype=torch.long, device=device)

    for i in range(E):
        u = pos_edge_index[0, i].item()
        count = 0
        while count < K:
            v = torch.randint(0, num_nodes, (1,), device=device).item()
            if u != v and (u, v) not in forbidden:
                neg_edges[i, count, 0] = u
                neg_edges[i, count, 1] = v
                count += 1

    return neg_edges


def load_ogbl(data_folder, dataset_name):

    dataset = PygLinkPropPredDataset(
        name=dataset_name,
        root=data_folder + "ogb"
    )
    ogb_data = dataset[0]

    # -------------------------
    # Node features
    # -------------------------
    ogb_data.x = ogb_data.x.to(torch.float)

    # L2 normalize features
    x_norm = ogb_data.x.norm(p=2, dim=1, keepdim=True)
    ogb_data.x = ogb_data.x / x_norm

    n_classes = dataset.num_classes

    split = dataset.get_edge_split()

    # ======================================================
    # PPA / COLLAB
    # ======================================================
    if dataset_name in ["ogbl-ppa", "ogbl-collab"]:

        # -------- TRAIN POS --------
        train_pos = split["train"]["edge"].T        # [2, E]
        train_pos = to_undirected(train_pos)
        self_loops = train_pos[0] == train_pos[1]
        train_pos = train_pos[:, ~self_loops]

        ogb_data.edge_index = train_pos
        ogb_data.ogb_train_pos_edge_index = train_pos

        # -------- TEST POS / NEG --------
        ogb_data.ogb_test_pos_edge_index = split["test"]["edge"].T   # [2, E]
        ogb_data.ogb_test_neg_edge_index = split["test"]["edge_neg"].T

    # ======================================================
    # CITATION2  (THIS IS THE IMPORTANT PART)
    # ======================================================
    elif dataset_name == "ogbl-citation2":

        # -------- TRAIN POS --------
        src = split["train"]["source_node"]
        dst = split["train"]["target_node"]

        train_pos = torch.stack([src, dst], dim=0)   # [2, E]
        train_pos = to_undirected(train_pos)
        self_loops = train_pos[0] == train_pos[1]
        train_pos = train_pos[:, ~self_loops]

        ogb_data.edge_index = train_pos
        ogb_data.ogb_train_pos_edge_index = train_pos

        # -------- TEST POS --------
        src = split["test"]["source_node"]
        dst = split["test"]["target_node"]
        ogb_data.ogb_test_pos_edge_index = torch.stack([src, dst], dim=0)

        # -------- CRITICAL: STORE RAW TEST STRUCTURE --------
        # These are REQUIRED later for correct MRR evaluation
        ogb_data.ogb_test_source_node = src                      # [N_pos]
        ogb_data.ogb_test_pos_target_node = dst                  # [N_pos]
        ogb_data.ogb_test_target_node_neg = split["test"]["target_node_neg"]  # [N_pos, N_neg]

    else:
        raise ValueError(f"Unsupported OGB dataset: {dataset_name}")

    return ogb_data, n_classes


# load dataset
def load_dataset(data_folder, dataset_name, rand_train=False):
        
    if dataset_name[:4] == 'ogbl':
        data, n_classes = load_ogbl(data_folder, dataset_name)
    else:
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ['Cornell', 'Wisconsin', 'Texas']:
            dataset = WebKB(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ['Squirrel', 'Chameleon']:
            dataset = WikipediaNetwork(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ['Crocodile']:
            dataset = WikipediaNetwork(root=f'{data_folder}pyG', name=dataset_name, 
                            geom_gcn_preprocess=False, transform=NormalizeFeatures())
        if dataset_name in ['Actor']:
            dataset = Actor(root=f'{data_folder}pyG/Actor', 
                        transform=NormalizeFeatures())
        if dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
            dataset = HeterophilousGraphDataset(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
    
        n_classes = dataset.num_classes 
        data = dataset[0]  # Get the first graph object.
        # convert to undirected graph
        data.edge_index = to_undirected(data.edge_index)
        # remove self loops 
        selfloops = data.edge_index[0,:]==data.edge_index[1,:]
        data.edge_index = data.edge_index[:,tf.logical_not(selfloops).numpy()]

    return(data, n_classes)

# needed for subgraphs where randomly selected nodes are relabelled for edge indices
def _filter_and_relabel(edge_index, selected_nodes, is_pyg, num_nodes):
    device = edge_index.device
    selected_nodes = selected_nodes.to(device)

    node_map = -torch.ones(
        num_nodes,
        dtype=torch.long,
        device=device
    )
    node_map[selected_nodes] = torch.arange(
        len(selected_nodes),
        device=device
    )

    if is_pyg:
        src, dst = edge_index
        mask = (node_map[src] >= 0) & (node_map[dst] >= 0)
        src = node_map[src[mask]]
        dst = node_map[dst[mask]]
        return torch.stack([src, dst], dim=0)

    else:
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        mask = (node_map[src] >= 0) & (node_map[dst] >= 0)
        src = node_map[src[mask]]
        dst = node_map[dst[mask]]
        return torch.stack([src, dst], dim=1)

def get_non_edge_index(n_nodes, edge_index):
    # generate random points of non-edges, same number of edges
    # initially taken double count to address removal due to:
        # self-loops & duplicates & overlap with edges
    n_edges = edge_index.shape[1]
    non_edge_index = np.random.randint(1, n_nodes, size=(2, int(n_edges*2.0)))
    valid_flags = non_edge_index[0]!=non_edge_index[1] # remove self-loops
    non_edge_index = non_edge_index[:, valid_flags]
    non_edge_index_set = set([tuple(t) for t in np.transpose(non_edge_index)])
    edge_index_set = set([tuple(t) for t in np.transpose(edge_index)])
    non_edge_index = list(non_edge_index_set - edge_index_set)
    non_edge_index = np.transpose(non_edge_index[:n_edges])
    #n_edges = non_edge_index.shape[0]
    # sort the index    
    lexsorted_indices = np.lexsort((non_edge_index[1, :], non_edge_index[0, :]))
    non_edge_index = non_edge_index[:, lexsorted_indices]
    return(torch.from_numpy(non_edge_index))

def remap_with_used_nodes(edge_index, selected_nodes, used_nodes):
    """
    edge_index: [2, E] (already filtered to selected_nodes)
    selected_nodes: original node ids in this subgraph
    used_nodes: indices (0..len(selected_nodes)-1) actually used by edges
    """
    device = edge_index.device

    # Map from selected-node index -> compact index
    node_map = -torch.ones(
        selected_nodes.numel(),
        dtype=torch.long,
        device=device
    )
    node_map[used_nodes] = torch.arange(
        used_nodes.numel(),
        device=device
    )

    return node_map[edge_index]

def get_subgraphs_ogb(dataset_name, data, max_nodes, max_parts=10):
    """
    SAFE OGB subgraphing:
    - edge-induced
    - no global adjacency
    """

    min_edges = 50
    max_tries = 10 * max_parts

    # --------------------------------------------------
    # Pool edges ONLY (train + test pos if available)
    # --------------------------------------------------
    edge_pool = [data.edge_index]

    if hasattr(data, "ogb_test_pos_edge_index"):
        edge_pool.append(data.ogb_test_pos_edge_index)

    edge_pool = torch.cat(edge_pool, dim=1)
    perm = torch.randperm(edge_pool.shape[1])

    subgraphs = []
    ptr = 0
    tries = 0

    while len(subgraphs) < max_parts and ptr < edge_pool.shape[1] and tries < max_tries:
        tries += 1

        # --------------------------------------------------
        # Edge-induced node sampling
        # --------------------------------------------------
        nodes = set()
        while ptr < edge_pool.shape[1] and len(nodes) < max_nodes:
            u, v = edge_pool[:, perm[ptr]].tolist()
            nodes.add(u)
            nodes.add(v)
            ptr += 1

        if len(nodes) < 2:
            continue

        selected_nodes = torch.tensor(sorted(nodes), dtype=torch.long)

        # --------------------------------------------------
        # Filter + relabel TRAIN POS
        # --------------------------------------------------
        train_pos = _filter_and_relabel(
            data.edge_index,
            selected_nodes,
            is_pyg=True,
            num_nodes=data.num_nodes
        )

        if train_pos.shape[1] < min_edges:
            continue

        # --------------------------------------------------
        # Remove isolated nodes (subgraph-local only)
        # --------------------------------------------------
        used_nodes = torch.unique(train_pos.flatten())

        if used_nodes.numel() < min(1000, max_nodes):
            continue

        # remap to compact IDs
        node_map = -torch.ones(selected_nodes.numel(), dtype=torch.long)
        node_map[used_nodes] = torch.arange(used_nodes.numel())

        train_pos = node_map[train_pos]

        x_sub = data.x[selected_nodes][used_nodes]

        sg = Data(
            x=x_sub,
            edge_index=train_pos,
            num_nodes=used_nodes.numel()
        )

        if hasattr(data, "y") and data.y is not None:
            sg.y = data.y[selected_nodes][used_nodes]

        # --------------------------------------------------
        # TRAIN negatives (subgraph-only, safe)
        # --------------------------------------------------
        sg.ogb_train_pos_edge_index = train_pos
        sg.ogb_train_neg_edge_index = get_non_edge_index(
            sg.num_nodes,
            train_pos
        )

        # --------------------------------------------------
        # TEST positives (if present)
        # --------------------------------------------------
        if hasattr(data, "ogb_test_pos_edge_index"):
            test_pos = _filter_and_relabel(
                data.ogb_test_pos_edge_index,
                selected_nodes,
                is_pyg=True,
                num_nodes=data.num_nodes
            )
            if test_pos.numel() > 0:
                sg.ogb_test_pos_edge_index = node_map[test_pos]

        # --------------------------------------------------
        # TEST negatives (if present)
        # --------------------------------------------------
        if hasattr(data, "ogb_test_neg_edge_index"):
            test_neg = _filter_and_relabel(
                data.ogb_test_neg_edge_index,
                selected_nodes,
                is_pyg=True,
                num_nodes=data.num_nodes
            )
            if test_neg.numel() > 0:
                sg.ogb_test_neg_edge_index = node_map[test_neg]

        # --------------------------------------------------
        # Citation2 grouped negatives (optional, safe)
        # --------------------------------------------------
        if hasattr(data, "ogb_test_source_node") and hasattr(data, "ogb_test_target_node_neg"):
            src = data.ogb_test_source_node
            tgt_neg = data.ogb_test_target_node_neg

            # map full graph nodes → subgraph nodes
            full_map = -torch.ones(data.num_nodes, dtype=torch.long)
            full_map[selected_nodes[used_nodes]] = torch.arange(used_nodes.numel())

            valid = full_map[src] >= 0
            sg.ogb_test_source_node = full_map[src[valid]]

            rows = []
            for row in tgt_neg:
                mapped = full_map[row]
                keep = mapped >= 0
                if keep.any():
                    rows.append(mapped[keep])

            sg.ogb_test_target_node_neg = rows

        subgraphs.append(sg)

    return subgraphs


            
def get_subgraphs_nonogb(dataset_name, data, max_nodes, max_parts=10):
    
    if dataset_name == "ogbl-ppa":
        min_edges = 100
    else:
        min_edges = 50

    n_nodes = data.num_nodes
    nodes = np.arange(n_nodes)
    np.random.shuffle(nodes)
    n_chunks = int(np.ceil(n_nodes/max_nodes))
    chunks = np.array_split(nodes, n_chunks)
    # remove last subgraph if too small
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks = chunks[:-1]
    if len(chunks) > max_parts:
        chunks = chunks[:max_parts]
    if len(chunks) < max_parts:
        chunks = (chunks*max_parts)[:max_parts]

    subgraphs = []
    chunk_count = 0 # may go upto max_parts
    for selected_nodes in chunks:
    
        selected_nodes = torch.tensor(sorted(selected_nodes), dtype=torch.long)
        
        # Extract the subgraph
        subgraph_edge_index = subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)[0]
        n_edges = subgraph_edge_index.shape[1]
        if n_edges < min_edges:
            continue
        subgraph_data = Data(x=data.x[selected_nodes], edge_index=subgraph_edge_index, num_nodes=len(selected_nodes))
        if hasattr(data, 'y'):
            if data.y is not None:
                subgraph_data = Data(x=data.x[selected_nodes], y=data.y[selected_nodes], edge_index=subgraph_edge_index, num_nodes=len(selected_nodes))
        subgraphs.append(subgraph_data)
        chunk_count += 1
        if chunk_count>= max_parts:
            break
    del data
    return(subgraphs)

def get_subgraphs(dataset_name, data, max_nodes, max_parts=10):
    """
    Dispatches to OGB or non-OGB splitting based on presence of edge_split.
    Signature unchanged.
    """
    if dataset_name.startswith("ogbl"):
        return get_subgraphs_ogb(dataset_name, data, max_nodes, max_parts=10)
    else:
        return get_subgraphs_nonogb(dataset_name, data, max_nodes, max_parts=10)

def ogb_num_test_negatives(dataset_name):
    if dataset_name == "ogbl-ppa":
        return 100
    if dataset_name == "ogbl-collab":
        return 50
    if dataset_name == "ogbl-citation2":
        return 1000
    raise ValueError(dataset_name)

def deduplicate_edge_index(edge_index):
    """
    edge_index: torch.Tensor [2, E]
    returns: torch.Tensor [2, E_unique]
    """
    # sort each edge (u,v) so undirected duplicates collapse
    u = torch.minimum(edge_index[0], edge_index[1])
    v = torch.maximum(edge_index[0], edge_index[1])
    edges = torch.stack([u, v], dim=0)

    # unique columns
    edges = torch.unique(edges, dim=1)
    return edges

def split_data_on_edges_ogb(dataset_name, data):

    if dataset_name == "ogbl-ppa":
        K_train, K_test = 3, 100
    elif dataset_name == "ogbl-collab":
        K_train, K_test = 3, 50
    else:
        K_train, K_test = 3, 100

    # build adjacency + degree
    adj = [set() for _ in range(data.num_nodes)]
    for u, v in data.ogb_train_pos_edge_index.t().tolist():
        adj[u].add(v)
        adj[v].add(u)

    degrees = torch.tensor([len(adj[i]) for i in range(data.num_nodes)])

    train_pos = data.ogb_train_pos_edge_index
    test_pos  = data.ogb_test_pos_edge_index
    train_pos = deduplicate_edge_index(train_pos)
    test_pos  = deduplicate_edge_index(test_pos)

    train_neg = sample_negatives_degree_distance(
        train_pos, data.num_nodes, adj, degrees, K_train
    )
    train_neg = deduplicate_edge_index(train_neg)

    test_neg = sample_negatives_degree_distance(
        test_pos, data.num_nodes, adj, degrees, K_test
    )
    test_neg = deduplicate_edge_index(test_neg)

    data_train = Data(
        x=data.x,
        edge_index=train_pos,
        pos_edge_index=train_pos,
        neg_edge_index=train_neg,
        num_nodes=data.num_nodes,
        num_features=data.x.size(1),
        num_edges=train_pos.size(1),
        n_pos=train_pos.size(1),
        n_neg=train_neg.size(1)
    )

    data_test = Data(
        x=data.x,
        edge_index=test_pos,
        pos_edge_index=test_pos,
        neg_edge_index=test_neg,
        num_nodes=data.num_nodes,
        num_features=data.x.size(1),
        num_edges=test_pos.size(1),
        n_pos=test_pos.size(1),
        n_neg=test_neg.size(1)
    )

    return data_train, data_test


def split_data_on_edges_nonogb(data, train_frac):
    
    def split_edges(edge_index, train_frac):
        n_edges = edge_index.shape[1]
        train_idx = np.random.choice(n_edges, int(train_frac*n_edges), replace=False)
    
        train_mask = np.zeros(n_edges, dtype=int)
        train_mask[train_idx] = 1
        train_mask = train_mask.astype(bool)
        test_mask = np.logical_not(train_mask)
    
        train_edge_index = edge_index[:, train_mask]
        test_edge_index = edge_index[:, test_mask]
        return(train_edge_index, test_edge_index)
    
    edge_index = data.edge_index
    n_nodes, n_features = data.num_nodes, data.num_features

    # generate non_edge_index using edge_index
    non_edge_index = get_non_edge_index(n_nodes, edge_index)

    if train_frac is None: # no split, but create data_train in right format
        
        n_edges = edge_index.shape[1]
        n_pos_edges = n_edges
        n_neg_edges = non_edge_index.shape[1]
        if hasattr(data, 'y'):
            data_train = Data(x=data.x, y=data.y, edge_index = edge_index,
                              pos_edge_index=edge_index, 
                              neg_edge_index=non_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_edges, 
                              n_pos=n_pos_edges, n_neg=n_neg_edges)
        else:
            data_train = Data(x=data.x, edge_index = edge_index,
                              pos_edge_index=edge_index, 
                              neg_edge_index=non_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_edges, 
                              n_pos=n_pos_edges, n_neg=n_neg_edges)
        data_test = None

    else:
        
        # split both edge and non-edge into train and test
        train_pos_edge_index, test_pos_edge_index = split_edges(data.edge_index, train_frac=train_frac)
        train_neg_edge_index, test_neg_edge_index = split_edges(non_edge_index, train_frac=train_frac)
        n_train_edges = train_pos_edge_index.shape[1]
        n_test_edges = test_pos_edge_index.shape[1]
        
        if hasattr(data, 'y'):
            data_train = Data(x=data.x, y=data.y, edge_index = train_pos_edge_index,
                              pos_edge_index=train_pos_edge_index, 
                              neg_edge_index=train_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_train_edges, 
                              n_pos=n_train_edges, n_neg=n_train_edges)
            data_test = Data(x=data.x, y=data.y, edge_index = test_pos_edge_index,
                              pos_edge_index=test_pos_edge_index, 
                              neg_edge_index=test_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_test_edges, 
                              n_pos=n_test_edges, n_neg=n_test_edges)
        else:
            data_train = Data(x=data.x, y=data.y, edge_index = train_pos_edge_index,
                              pos_edge_index=train_pos_edge_index, 
                              neg_edge_index=train_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_train_edges, 
                              n_pos=n_train_edges, n_neg=n_train_edges)
            data_test = Data(x=data.x, y=data.y, edge_index = test_pos_edge_index,
                              pos_edge_index=test_pos_edge_index, 
                              neg_edge_index=test_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_test_edges, 
                              n_pos=n_test_edges, n_neg=n_test_edges)
            
    return(data_train, data_test)


def split_data_on_edges(dataset_name, data, train_frac=None):
    """
    Dispatches to OGB or non-OGB splitting based on presence of edge_split.
    Signature unchanged.
    """
    if dataset_name.startswith("ogbl"):
        return split_data_on_edges_ogb(dataset_name, data)
    else:
        return split_data_on_edges_nonogb(data, train_frac=train_frac)
