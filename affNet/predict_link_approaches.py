"""
program: Compare link prediction and learning curve for 4 approaches
version: sparse tf version, multi-headed affinity with separator learning
author: indranil ojha

"""

import os, sys
if "spyder_kernels" in sys.modules:
    env = 'IDE' # IDE (run segment wise within spyder) or CMD (run whole script from commandline)
else:
    env = 'CMD'

# setup environment
root = 'D:/Indranil/JRF/Submission/affinity/codebase/affNet/'
data_folder = "D:/Indranil/ML2/Datasets/"

# import libraries, including utils
sys.path.append(root)

import pandas as pd
import numpy as np
import time
from utils import set_seeds, get_params, get_edge_h, eval_link_pred
from utils import plot_hist, parse_arg
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

results_fname = results_folder + "approaches.csv"
result_cols = ['dataset', 'nodes', 'features', 'classes', 'directed', 
               'emb_features', 'n_heads', 'max_nodes', 'init_lr', 'train_frac', 'epochs', 
               'edge_h', 'aff_h', 'sep_learning', 'beta', 'metric_name', 'metric_value', 
               'elapsed', 'seed']

try:
   results_df = pd.read_csv(results_fname) 
except:
    results_df = pd.DataFrame(columns = result_cols)

if env=='CMD':
    dataset_name, emb_features, n_heads, max_nodes, init_lr, epochs = parse_arg(root) # arguments passed thru commandline
    datasets = [dataset_name]
else:
    datasets = ['Texas', 'Wisconsin', 'Cora', 'CiteSeer', 'Photo', 'Chameleon', 'Squirrel',
                'ogbl-ppa', 'ogbl-collab', 'ogbl-citation2']

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
    n_orig_nodes = n_nodes
    print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')

    data = get_random_subgraph(data, max_nodes)
    data_train, data_test = split_data_on_edges(data, train_frac=train_frac) 
    #data_train = convert_data_to_tf(data_train)
    data_train = tf_GData(data_train)
    data_test = tf_GData(data_test)
    n_nodes, n_edges = data_train.num_nodes, data_train.num_edges

    loss_histories, aff_histories, metric_histories, aff_matrices, aff_values = [], [], [], [], []

    # options holds values of no of heads and if sep_learing should be on - 4 combinations
    options = [(1, False), (n_heads, False), (1, True),(n_heads, True)]
    for heads, sep_learning in options:
        print(f"running for {heads} head(s) {'with' if sep_learning else 'without'} separator learning")
        start = time.time()
    
        aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric = compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)
        metric_name, metric_value, auc = eval_link_pred(dataset_name, aff_matrix, data_test.pos_edge_index, data_test.neg_edge_index)
        if dataset_name[:4] != 'ogbl':
            metric_name, metric_value = 'AUC', auc
        
        print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, beta={beta:.2f}, metric value: {metric_value: .4f}")
        
        #affinity_model.save(f'{dataset_name}_model.h5')
    
        loss_histories.append(hist_loss)
        aff_histories.append(hist_aff)
        metric_histories.append(hist_metric)
        aff_matrices.append(aff_matrix)
        stop = time.time()
        elapsed = int(stop-start)

        results_df.loc[len(results_df)] = [dataset_name, n_orig_nodes, n_features, n_classes, 
                is_directed, emb_features, heads, max_nodes, init_lr, train_frac,
                epochs, edge_h, aff_h, sep_learning, beta, metric_name, metric_value, elapsed, seed]
        results_df.to_csv(results_fname, index=False, float_format="%.4f")

    aff_matrices = [data_train.pos_edge_index] + aff_matrices
    gc.collect()

    # save results
    plot_flag, save_flag = False, True
    plot_hist(loss_histories, dataset_name, results_folder, "Loss", plot_flag, save_flag)

