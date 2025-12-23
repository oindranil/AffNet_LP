"""
program: utilities needed by affinity matrix computation script
author: indranil ojha

"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.random import set_seed as tf_seed
from torch_geometric.utils import remove_isolated_nodes, homophily 
from numpy.random import seed as np_seed
from random import seed as random_seed
from sklearn.metrics import roc_auc_score
from ogb.linkproppred import Evaluator
import torch
import numpy as np
import os

# parse command line arguments
def parse_arg(root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--emb_features', type=int)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--max_nodes', type=int, default=13)
    parser.add_argument('--init_lr', type=float)
    parser.add_argument('--epochs', type=int)

    args = parser.parse_args()
    
    dataset_name = args.dataset
    emb_features = args.emb_features
    n_heads = args.n_heads
    max_nodes = args.max_nodes
    init_lr = args.init_lr
    epochs = args.epochs

    return(dataset_name, emb_features, n_heads, max_nodes, init_lr, epochs)

# read saved parameters for dataset
def get_params(dataset_name, root):
    params_fname = f"{root}params/params.csv"
    params_df = pd.read_csv(params_fname)
    params_df = params_df[params_df['dataset']==dataset_name]
    params_df = params_df[['emb_features', 'n_heads', 'max_nodes', 'pca_preserve', 'init_lr', 'epochs']]
    emb_features, n_heads, max_nodes, pca_preserve, init_lr, epochs = params_df.iloc[0].values
    emb_features = int(emb_features)
    n_heads = int(n_heads)
    max_nodes = int(max_nodes)
    epochs = int(epochs)
    return(emb_features, n_heads, max_nodes, init_lr, epochs)

# set all seeds for reproducibility
def set_seeds(seed=13):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random_seed(seed)
    tf_seed(seed)
    np_seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device == 'cuda:0':
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    
# retrieve homophily for dataset
def get_edge_h(dataset_name, root):
    edge_h_fname = f"{root}params/edge_h.csv"
    edge_h_df = pd.read_csv(edge_h_fname)
    edge_h_df = edge_h_df[edge_h_df['dataset']==dataset_name]
    edge_h_df = edge_h_df['edge_h']
    return(edge_h_df.iloc[0])

# compute node level edge homophily
def get_node_h(adj, y):
    eps = 1e-8
    n_nodes = len(y)
    y_np = y.numpy()
    # adj can be sparse or dense
    if isinstance(adj, tf.Tensor): # dense
        adj_np = adj.numpy()
    else: # sparse
        adj_np = tf.sparse.to_dense(adj).numpy()
    yy = np.tile(y_np, (n_nodes,1))
    yy_t = np.transpose(yy)
    match = (yy==yy_t).astype(int) 
    match = np.multiply(adj_np, match)
    nodewise_match = np.sum(match, axis=1)
    degree = np.sum(adj_np, axis=1)
    node_h = np.divide(nodewise_match+eps, degree+eps)
    return(node_h)    

def get_self_affinity(aff_matrix):
    n_nodes = aff_matrix.shape[0]
    diag = tf.linalg.diag_part(aff_matrix).numpy()
    all_sum = tf.reduce_sum(aff_matrix).numpy()
    diag_sum = tf.reduce_sum(diag).numpy()
    non_diag_sum = all_sum - diag_sum
    non_diag_mean = non_diag_sum/(n_nodes * (n_nodes - 1))
    self_affinity = diag / non_diag_mean
    return(self_affinity)

def get_degree_1D(edge_index, n_nodes):
    deg = np.zeros(n_nodes)
    for i in range(n_nodes):
        deg[i] = np.sum(edge_index.numpy()[0,:]==i)
    return(deg)

def remove_isolated(data):
    n_nodes_before =data.x.shape[0]
    _, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=n_nodes_before)
    data = data.subgraph(mask)
    n_nodes_after =data.x.shape[0]
    nodes_removed = n_nodes_before - n_nodes_after
    return(data, nodes_removed)

def get_homophily(data):
    h = homophily(data.edge_index, data.y, method='edge')
    return(h)

def bin_it(a, n_bins=5):
    min_a, max_a = np.min(a), np.max(a)
    width = (max_a-min_a)/n_bins
    b = np.floor(((a-min_a)/width))
    return(b)

def plot_hist(losses, dataset_name, results_folder, ylabel, plot_flag=True, save_flag=False):
    epochs = len(losses[0])
    title = dataset_name
    legends = ["single-headed w/o sep", "multi-headed w/o sep", 
               "single-headed with sep", "multi-headed with sep"]
    linestyles = ['dotted', 'dashed','solid', 'solid' ]
    linewidths = [2, 2, 2, 3]
    plt.figure(figsize=(6,6), dpi=600)
    for i, loss in enumerate(losses):
        plt.plot(loss, label=legends[i], linestyle=linestyles[i], linewidth=linewidths[i])
    plt.xlabel("epochs", fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.title(title, fontsize=36)
    plt.xticks([0, epochs//2, epochs], [0, epochs//2, epochs], fontsize=28)
    plt.yticks([0, 0.2, 0.4], [0, 0.2, 0.4], fontsize=28)
    #plt.yticks(fontsize=28)
    plt.tight_layout()
    if save_flag:
        savefile=f"{results_folder}hist/{dataset_name} {ylabel}.png"
        plt.savefig(savefile)
    if plot_flag:
        plt.show()

def plot_affinity(aff_matrices, dataset_name, results_folder, plot_flag=True, save_flag=False, n_points=100):
    adj = tf.sparse.to_dense(aff_matrices[0])[:n_points,:n_points]
    aff_matrices = [adj, aff_matrices[-1]]

    # plot affinity matrix against adjacency
    c00, c01, c1 = 'ivory', 'navajowhite', 'darkgoldenrod'
    plt.figure(figsize=(6,3), dpi=600)   
    n_plots = len(aff_matrices)
    for k in range(n_plots):
        if k==0:
            c0 = c00
        else:
            c0 = c01
        aff_mat = aff_matrices[k][:n_points,:n_points]
        aff_mat = aff_mat.numpy()
        zeros = np.where(aff_mat<0.5)
        ones = np.where(aff_mat>=0.5)
        plt.subplot(1, n_plots, k+1)
        plt.scatter(zeros[0], n_points-zeros[1], color=c0, s=5)
        plt.scatter(ones[0], n_points-ones[1], color=c1, s=5)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color('black')  # Set boundary color to blue
            #spine.set_linewidth(2)   # Optional: Set the boundary thickness
    plt.tight_layout()
    if save_flag:
        savefile=f"{results_folder}aff_plots/{dataset_name}.png"
        plt.savefig(savefile)
    if plot_flag:
        plt.show()

# compute hit@k between true and pred 
def compute_hit_rate(true, pred, k=50):
    # hit rate for edges or positive samples
    sort_idx = np.argsort(pred)[::-1]
    t = true[sort_idx][:k]
    hit = np.sum(t)
    hit_at_k = hit / k
   
    return(hit_at_k)

def eval_citation2_mrr(aff, data_test):
    """
    Manual MRR for ogbl-citation2.
    Uses existing stored edges.
    """

    # aff: numpy [N, N]
    pos = data_test.pos_edge_index          # [2, E_pos]
    src = pos[0].cpu().numpy()
    tgt_pos = pos[1].cpu().numpy()

    tgt_neg_list = data_test.ogb_test_target_node_neg
    # ^ list of 1D torch tensors (ragged)

    mrr_list = []

    for i in range(5):
        s = src[i]
        neg_t = tgt_neg_list[i]
        print(
            f"i={i}, src={s}, "
            f"neg_min={neg_t.min().item()}, "
            f"neg_max={neg_t.max().item()}, "
            f"count={len(neg_t)}"
        )
    
    for i in range(len(src)):
        s = src[i]
        p = tgt_pos[i]

        pos_score = aff[s, p]

        neg_t = tgt_neg_list[i].cpu().numpy()
        neg_scores = aff[s, neg_t]

        rank = 1 + np.sum(neg_scores >= pos_score)
        mrr_list.append(1.0 / rank)

    return "MRR", float(np.mean(mrr_list))

def eval_link_pred_ogb(dataset_name, aff, data_test):

    if tf.is_tensor(aff):
        aff = aff.numpy()

    pos = data_test.pos_edge_index.cpu().numpy()
    neg = data_test.neg_edge_index.cpu().numpy()

    E_pos = pos.shape[1]
    K = neg.shape[1] // E_pos

    hits = []
    mrrs = []

    for i in range(E_pos):
        u, v = pos[:, i]
        s_pos = aff[u, v]

        neg_scores = []
        for j in range(K):
            idx = i * K + j
            nu, nv = neg[:, idx]
            neg_scores.append(aff[nu, nv])

        neg_scores = np.array(neg_scores)
        rank = 1 + np.sum(neg_scores > s_pos)

        if dataset_name == "ogbl-collab":
            hits.append(rank <= 50)
        else:
            hits.append(rank <= 100)

        mrrs.append(1.0 / rank)

    if dataset_name == "ogbl-citation2":
        return "MRR", float(np.mean(mrrs))

    Kname = "Hits@50" if dataset_name == "ogbl-collab" else "Hits@100"
    return Kname, float(np.mean(hits))


def eval_link_pred_nonogb(dataset_name, aff, data_test):
    """
    Non-OGB evaluation using AUROC.
    Assumes:
      - pos_edge_index, neg_edge_index are torch tensors [2, E]
      - aff is tf.Tensor or numpy [N, N]
    """

    # affinity → numpy
    if tf.is_tensor(aff):
        aff = aff.numpy()

    pos = data_test.pos_edge_index   # [2, E_pos]
    neg = data_test.neg_edge_index   # [2, E_neg]

    assert isinstance(pos, torch.Tensor)
    assert isinstance(neg, torch.Tensor)

    pos = pos.cpu().numpy()
    neg = neg.cpu().numpy()

    # scores
    y_pos = aff[pos[0], pos[1]]
    y_neg = aff[neg[0], neg[1]]

    # labels
    y_true = np.concatenate([
        np.ones(len(y_pos)),
        np.zeros(len(y_neg))
    ])

    y_score = np.concatenate([y_pos, y_neg])

    auc = roc_auc_score(y_true, y_score)

    return "AUC", auc

def eval_link_pred(dataset_name, aff, data_test):
    if dataset_name.startswith("ogbl"):
        return eval_link_pred_nonogb(dataset_name, aff, data_test)
    else:
        return eval_link_pred_nonogb(dataset_name, aff, data_test)

def get_structure_info(data):

    n_nodes = data.x.shape[0]
    n_edges = data.edge_index.shape[1]

    # avg degree 
    avg_degree = n_edges/n_nodes

    # no of nodes with zero degree 
    active_nodes = len(torch.unique(data.edge_index))
    isolated_frac = (n_nodes - active_nodes) / n_nodes
    
    return(avg_degree, isolated_frac)


