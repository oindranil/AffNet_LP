"""
program: compute affinity matrix for datasets 
version: sparse tf version, multi-headed affinity with separator learning
previous version: v6
    changes from previous: primarily, more modular
        a. model training and affinity matrix computation taken out as 
            compute _afffinity which is put under models.py script
        b. separate scripts written for homophily, link prediction and trend
        c. torch to tf conversion for Data has been done thru python class instead of PyG

author: indranil ojha

"""

# setup environment
env = 'windows' # ubuntu or windows
if env=='windows':
   	root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/codebase/affNet/'
   	data_folder = "D:/Indranil/ML2/Datasets/"

# import libraries, including utils
import os, sys
sys.path.append(root)

import pandas as pd
import numpy as np
import time
from utils import set_seeds, get_params, get_edge_h, eval_link_pred
from utils import plot_affinity, plot_hist
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
if not os.path.exists(results_folder+'hist'):
    os.makedirs(results_folder+'hist')

datasets = ['Texas', 'Cornell', 'Wisconsin', 'Actor', 'Cora', 'CiteSeer']

print()
for dataset_name in datasets:

    set_seeds(seed)

    # read edge homophily value for dataset, for reporting
    edge_h = get_edge_h(dataset_name, root)    

    # get best parameters
    emb_features, n_heads, max_nodes, init_lr, epochs = get_params(dataset_name, root)
    min_lr, decay_steps = 0.0001, 25
    lr_decay = np.power((min_lr/init_lr), decay_steps/epochs)
    train_frac=0.8

    # load dataset
    data, n_classes = load_dataset(data_folder, dataset_name)
    data.num_features = data.x.shape[1]
    n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
    print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')

    data = get_random_subgraph(data, max_nodes)
    data_train, data_test = split_data_on_edges(data, train_frac=train_frac) 
    #data_train = convert_data_to_tf(data_train)
    data_train = tf_GData(data_train)
    data_test = tf_GData(data_test)
    n_nodes, n_edges = data_train.num_nodes, data_train.num_edges

    loss_histories, aff_histories, metric_histories, aff_matrices, aff_values = [], [], [], [], []

    # options holds values of no of heads and if sep_learing should be on - 4 combinations
    options = [(1, False), (n_heads, False), (1, True), (n_heads, True)]
    for heads, sep_learning in options:
        print(f"running for {heads} head(s) {'with' if sep_learning else 'without'} separator learning")
        start = time.time()
    
        aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric = compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)
        
        print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, beta={beta:.2f}")
        
        #affinity_model.save(f'{dataset_name}_model.h5')
    
        loss_histories.append(hist_loss)
        aff_histories.append(hist_aff)
        metric_histories.append(hist_metric)
        aff_matrices.append(aff_matrix)
        stop = time.time()
        elapsed = int(stop-start)

    aff_matrices = [data_train.pos_edge_index] + aff_matrices
    gc.collect()

    # save results
    np.savetxt(f"{results_folder}hist/hist_{dataset_name}.csv", np.array(loss_histories), delimiter=',', fmt='%.4f')
    if env=="windows":
        plot_flag, save_flag = True, True
    else:
        plot_flag, save_flag = False, True
    plot_hist(loss_histories, dataset_name, results_folder, "Loss", plot_flag, save_flag)
