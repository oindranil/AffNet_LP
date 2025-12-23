"""
program: AffNet - affinity plots
version: first version

"""

# setup environment
env = 'windows' 
if env=='windows':
	root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/codebase/affNet/'
	data_folder = "D:/Indranil/ML2/Datasets/"

# import libraries, including utils
import os, sys
sys.path.append(root)

import numpy as np
import time
from utils import set_seeds, get_params, get_edge_h, eval_link_pred
from utils import plot_affinity
from models import compute_affinty
from load import load_dataset, split_data_on_edges, get_random_subgraph, tf_GData
import gc
    
###
### main program ###
###

seed = 13
set_seeds(seed)

results_folder = f"{root}results/"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if not os.path.exists(results_folder+'aff_plots'):
    os.makedirs(results_folder+'aff_plots')

datasets = ['Texas', 'Cora', 'Wisconsin', 'CiteSeer']

print()
for dataset_name in datasets:

    set_seeds(seed)

        # read edge homophily value for dataset, for reporting
    edge_h = get_edge_h(dataset_name, root)    

    # get best parameters
    emb_features, n_heads, max_nodes, init_lr, epochs = get_params(dataset_name, root)
    min_lr, decay_steps = 0.0001, 25
    lr_decay = np.power((min_lr/init_lr), decay_steps/epochs)
    heads, sep_learning = n_heads, True
    train_frac=0.8
    max_parts = 10

    # load dataset
    data, n_classes = load_dataset(data_folder, dataset_name)
    #data.x = apply_pca(data.x, pca_preserve)
    data.num_features = data.x.shape[1]
    n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
    n_orig_nodes = n_nodes
    print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')

    sg = get_random_subgraph(data, max_nodes)
    del data

    data_train, data_test = split_data_on_edges(sg, train_frac=train_frac) 
    data_train = tf_GData(data_train)
    data_test = tf_GData(data_test)
    n_nodes, n_edges = data_train.num_nodes, data_train.num_edges    

    start = time.time()

    aff_matrix, aff_h, beta, hist_loss, _, _ = compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)    
    metric_name, metric_value, auc = eval_link_pred(dataset_name, aff_matrix, data_test.pos_edge_index, data_test.neg_edge_index)
    print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, beta={beta:.2f}, {metric_name}: {metric_value:.4f}")
    
    stop = time.time()
    elapsed = int(stop-start)
    gc.collect()

    aff_matrices = [data_train.pos_edge_index, aff_matrix] 

    # save results
    if env=="windows":
        plot_flag, save_flag = True, True
    else:
        plot_flag, save_flag = False, True
        
    plot_affinity(aff_matrices, dataset_name, results_folder, plot_flag, save_flag)




