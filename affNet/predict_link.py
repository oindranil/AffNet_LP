"""
program: AffNet - homophily estimation and link prediction
version 4: taken from v2, implemented ogb splits
This is a version that runs all datasets without error. But results for OGB are poor, so next version is required.
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
    	root = 'D:/Indranil/JRF/Submission/Affinity/codebase5/affNet_LP/affNet/'
    	data_folder = "D:/Indranil/ML2/Datasets/"
    elif system=='linux':
        root = "/home/iplab/indro/ml2/affinity/affNet_4/"
        data_folder = "/home/iplab/indro/ml2/Datasets/"

if system=='windows':
    data_folder = "D:/Indranil/ML2/Datasets/"
elif system=='linux':
    data_folder = "/home/iplab/indro/ml2/Datasets/"

sys.path.append(root)


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import set_seeds, get_params, get_edge_h, eval_link_pred, parse_arg
#from utils import plot_affinity
from models import compute_affinty
from load import load_dataset, split_data_on_edges, get_subgraphs, tf_GData
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
if not os.path.exists(results_folder+'hist'):
    os.makedirs(results_folder+'hist')

result_cols = ['dataset', 'nodes', 'features', 'classes', 'directed', 
               'emb_features', 'n_heads', 'max_nodes', 'init_lr', 'train_frac', 'epochs', 
               'n_train_pos', 'n_train_neg', 'n_test_pos', 'n_test_neg',
               'edge_h', 'aff_h', 'min_beta', 'max_beta', 'metric_name', 
               'metric_value', 'metric_std', 'elapsed', 'seed']

try:
   results_df = pd.read_csv(results_folder+'results.csv') 
except:
    results_df = pd.DataFrame(columns = result_cols)

# set up basic parameters
if env=='CMD':
    dataset_name, emb_features, n_heads, max_nodes, init_lr, epochs = parse_arg(root) # arguments passed thru commandline
    datasets = [dataset_name]
else:
    datasets = ['Texas', 'Wisconsin', 'Cora', 'CiteSeer', 'Photo', 'Chameleon', 'Squirrel',
                'ogbl-ppa', 'ogbl-collab', 'ogbl-citation2']
#    datasets = ['ogbl-citation2']
    datasets = ['Texas']


min_lr, decay_steps = 0.0001, 25
train_frac=0.8
max_parts = 5

print()
for dataset_name in datasets:

    set_seeds(seed)

    # read edge homophily value for dataset, for reporting
    edge_h = get_edge_h(dataset_name, root)    

    # get best parameters
    if env=='IDE':
        emb_features, n_heads, max_nodes, init_lr, epochs = get_params(dataset_name, root)
    heads, sep_learning = n_heads, True
    lr_decay = np.power((min_lr/init_lr), decay_steps/epochs)

    # load dataset
    data, n_classes = load_dataset(data_folder, dataset_name)
    data.num_features = data.x.shape[1]
    n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
    n_orig_nodes = n_nodes
    print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5}')
    
    subgraphs = get_subgraphs(dataset_name, data, max_nodes, max_parts)
    del data
    n_chunks = len(subgraphs)
    aff_matrices, aff_values, beta_values, elapsed_times = [], [], [], []
    metrics = []
    n_train_pos_list, n_train_neg_list, n_test_pos_list, n_test_neg_list = [], [], [], []
    i = 0
    for sg in subgraphs[:max_parts]:

        i += 1
        print(f'chunk {i}/{max_parts}')
        data_train, data_test = split_data_on_edges(dataset_name, sg, train_frac=train_frac) 
        n_train_pos, n_train_neg = data_train.n_pos, data_train.n_neg 
        n_test_pos, n_test_neg = data_test.n_pos, data_test.n_neg 
        n_train_pos_list.append(data_train.n_pos)
        n_train_neg_list.append(data_train.n_neg)
        n_test_pos_list.append(data_test.n_pos)
        n_test_neg_list.append(data_test.n_neg)
            
        tf_train = tf_GData(data_train)
        tf_test = tf_GData(data_test)
        n_nodes, n_edges = data_train.num_nodes, data_train.num_edges    
    
        start = time.time()
    
        aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric = compute_affinty(dataset_name, tf_train, tf_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)    
        metric_name, metric_value = eval_link_pred(dataset_name, aff_matrix, data_test)
        print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, beta={beta:.2f}, {metric_name}: {metric_value:.4f}")
        if i==1:
            plt.plot(hist_loss)
            plt.title(dataset_name)
            savefile=f"{results_folder}hist/{dataset_name}.png"
            plt.savefig(savefile)
            if env=='IDE':
                plt.show()
            else:
                plt.close()
        
        aff_matrices.append(aff_matrix)
        beta_values.append(beta)
        aff_values.append(aff_h)
        metrics.append(metric_value)
        stop = time.time()
        elapsed = int(stop-start)
        elapsed_times.append(elapsed)
        gc.collect()

    n_train_pos, n_train_neg = np.mean(n_train_pos_list), np.mean(n_train_neg_list)
    n_test_pos, n_test_neg = np.mean(n_test_pos_list), np.mean(n_test_neg_list)
    aff_matrices = [data_train.pos_edge_index] + aff_matrices
    aff_h = np.mean(aff_values)
    min_beta = np.min(beta_values)
    max_beta = np.max(beta_values)
    metric_value = np.mean(np.array(metrics), axis=0)
    metric_std = np.std(np.array(metrics), axis=0)
    elapsed = np.mean(elapsed_times)

    # save results
    if env=="windows":
        plot_flag, save_flag = True, True
    else:
        plot_flag, save_flag = False, True
        
    #plot_affinity(aff_matrices, dataset_name, results_folder, plot_flag, save_flag)

    print(f"n_train_pos:{n_train_pos}, n_train_neg:{n_train_neg}, n_test_pos:{n_test_pos}, n_test_neg:{n_test_neg}")
    print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, min_beta={min_beta:.2f}, max_beta={max_beta:.2f}, {metric_name}: {metric_value:.4f}")
    results_df.loc[len(results_df)] = [dataset_name, n_orig_nodes, n_features, n_classes, 
            is_directed, emb_features, n_heads, max_nodes, init_lr, train_frac,
            epochs, n_train_pos, n_train_neg, n_test_pos, n_test_neg, 
            edge_h, aff_h, min_beta, max_beta, metric_name, 
            metric_value, metric_std, elapsed, seed]
    results_df.to_csv(results_folder+'results.csv', index=False, float_format="%.4f")



