"""
program: analyze variation of beta across disjoint subgraphs 
   
"""

# setup environment
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os, sys
# setup environment
system = 'windows' # windows or linux
PF = 'torch' # platform torch 

if "spyder_kernels" in sys.modules:
    env = 'IDE' # IDE (run segment wise within spyder) or CMD (run whole script from commandline)
else:
    env = 'CMD'

if env=='CMD':
    root = os.getcwd()+'/'
else:
    if system=='windows':
    	root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/codebase/affNet/'
    	data_folder = "D:/Indranil/ML2/Datasets/"
    elif system=='linux':
        root = "/home/iplab/indro/ml2/affinity/affNet/affNet_4/"
        data_folder = "/home/iplab/indro/ml2/Datasets/"

if system=='windows':
    data_folder = "D:/Indranil/ML2/Datasets/"
elif system=='linux':
    data_folder = "/home/iplab/indro/ml2/Datasets/"

sys.path.append(root)

import pandas as pd
import numpy as np
import time
from utils import set_seeds, get_params, parse_arg, get_edge_h, eval_link_pred
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

#datasets = ["Cora", "CiteSeer", "PubMed"]
#datasets = ["ogbl-ppa", "ogbl-collab"]
#datasets = ["ogbl-citation2"]
if env=='CMD':
    dataset_name, emb_features, n_heads, max_nodes, init_lr, epochs = parse_arg(root) # arguments passed thru commandline
    datasets = [dataset_name]
else:
    datasets = ['Cora']

min_lr, decay_steps = 0.0001, 25
train_frac=0.8
max_parts = 5

result_cols = ['dataset', 'run_no', 'edge_h', 'aff_h', 'beta', 'metric_name', 'metric_value']

try:
   results_df = pd.read_csv(results_folder+'beta_values.csv') 
except:
    results_df = pd.DataFrame(columns = result_cols)

print()
for dataset_name in datasets:

    set_seeds(seed)

        # read edge homophily value for dataset, for reporting
    edge_h = get_edge_h(dataset_name, root)    

    # get best parameters
    emb_features, n_heads, max_nodes, init_lr, epochs = get_params(dataset_name, root)

    lr_decay = np.power((min_lr/init_lr), decay_steps/epochs)
    heads, sep_learning = n_heads, True

    # load dataset
    data, n_classes = load_dataset(data_folder, dataset_name)
    #data.x = apply_pca(data.x, pca_preserve)
    data.num_features = data.x.shape[1]
    n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
    n_orig_nodes = n_nodes
    print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')

    results = []
    run_no = 0
    for run_no in range(max_parts):

        print(f'chunk {run_no+1}/{max_parts}')
        sg = get_random_subgraph(data, max_nodes)
        data_train, data_test = split_data_on_edges(sg, train_frac=train_frac) 
        data_train = tf_GData(data_train)
        data_test = tf_GData(data_test)
        n_nodes, n_edges = data_train.num_nodes, data_train.num_edges    
    
        start = time.time()
    
        aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric = compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)    
        metric_name, metric_value, auc = eval_link_pred(dataset_name, aff_matrix, data_test.pos_edge_index, data_test.neg_edge_index)
        print(f"run: {run_no:>3}: edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, beta={beta:.2f}, {metric_name}: {metric_value:.4f}")
        
        results.append([dataset_name, run_no, edge_h, aff_h, beta, metric_name, metric_value])
        #beta_values.append(beta)
        #aff_values.append(aff_h)
        #metrics.append(metric_value)
        stop = time.time()
        elapsed = int(stop-start)
        #elapsed_times.append(elapsed)
        del aff_matrix
        gc.collect()

    del data

    # save results
    new_results = pd.DataFrame(results, columns=result_cols)    
    if results_df.empty:
        results_df = new_results
    else:
        results_df = pd.concat([results_df, new_results], ignore_index=True)
    results_df.to_csv(results_folder+'beta_values.csv', index=False, float_format="%.4f")



