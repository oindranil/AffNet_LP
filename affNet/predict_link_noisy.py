"""
program: AffNet 

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
import matplotlib.pyplot as plt
from utils import set_seeds, get_params, get_edge_h, eval_link_pred, parse_arg
from models import compute_affinty
from load import load_dataset, split_data_on_edges, get_subgraphs, tf_GData
from noise import apply_feature_noise
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
               'edge_h', 'aff_h', 'min_beta', 'max_beta', 'metric_name', 
               'metric_value', 'metric_std', 'auc_value', 'auc_std', 'noise_std', 'noise_drop',
               'elapsed', 'seed']

try:
   results_df = pd.read_csv(results_folder+'results_noise.csv') 
except:
    results_df = pd.DataFrame(columns = result_cols)

# set up basic parameters
if env=='CMD':
    dataset_name, emb_features, n_heads, max_nodes, init_lr, epochs = parse_arg(root) # arguments passed thru commandline
    datasets = [dataset_name]
else:
    datasets = ['Texas', 'Wisconsin', 'Cora', 'CiteSeer', 'Photo', 'Squirrel', 'Chameleon',
                'ogbl-ppa', 'ogbl-collab', 'ogbl-citation2']

min_lr, decay_steps = 0.0001, 25
train_frac=0.8
max_parts = 5
noise_drop = 0.05

print()
for dataset_name in datasets:
    for noise_std in [0.05, 0.1, 0.2]:

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
        # apply noise
        data.x = apply_feature_noise(data.x, gaussian_std=noise_std, dropout_prob=noise_drop)
        data.num_features = data.x.shape[1]
        n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
        n_orig_nodes = n_nodes
        print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5}')
    
        subgraphs = get_subgraphs(dataset_name, data, max_nodes, max_parts)
        del data
        n_chunks = len(subgraphs)
        aff_matrices, aff_values, beta_values, elapsed_times = [], [], [], []
        metrics, auc_values = [], []
        i = 0
        for sg in subgraphs:
    
            i += 1
            print(f'chunk {i}/{n_chunks}')
            data_train, data_test = split_data_on_edges(sg, train_frac=train_frac) 
            data_train = tf_GData(data_train)
            data_test = tf_GData(data_test)
            n_nodes, n_edges = data_train.num_nodes, data_train.num_edges    
        
            start = time.time()
        
            aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric = compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)    
            metric_name, metric_value, auc = eval_link_pred(dataset_name, aff_matrix, data_test.pos_edge_index, data_test.neg_edge_index)
            print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, beta={beta:.2f}, auc={auc:.2f}, {metric_name}: {metric_value:.4f}")
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
            auc_values.append(auc)
            stop = time.time()
            elapsed = int(stop-start)
            elapsed_times.append(elapsed)
            gc.collect()
    
        aff_matrices = [data_train.pos_edge_index] + aff_matrices
        aff_h = np.mean(aff_values)
        min_beta = np.min(beta_values)
        max_beta = np.max(beta_values)
        metric_value = np.mean(np.array(metrics), axis=0)
        auc_value = np.mean(np.array(auc_values), axis=0)
        metric_std = np.std(np.array(metrics), axis=0)
        auc_std = np.std(np.array(auc_values), axis=0)
        elapsed = np.mean(elapsed_times)
    
        print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, noise_std: {noise_std:.2f}, {metric_name}: {metric_value:.4f}, auc: {auc:.2f}")
        results_df.loc[len(results_df)] = [dataset_name, n_orig_nodes, n_features, n_classes, 
                is_directed, emb_features, n_heads, max_nodes, init_lr, train_frac,
                epochs, edge_h, aff_h, min_beta, max_beta, metric_name, 
                metric_value, metric_std, auc_value, auc_std, noise_std, noise_drop, elapsed, seed]
        results_df.to_csv(results_folder+'results_noise.csv', index=False, float_format="%.4f")



