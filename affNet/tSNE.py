"""
program: AffNet: tSNE plotting 

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
import tensorflow as tf
from utils import set_seeds, get_params
from models import compute_affinty2
from load import load_dataset, split_data_on_edges, get_subgraphs, tf_GData
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import gc

def plot_in_2D(root, method, purpose, dataset_name, X, y, n_components=2):
    # purpose can be "labels" or "heads"
    # method canbe "tSNE" or "PCA"
    colors = ['blue', 'red', 'green', 'brown', 'yellow', 'magenta', 'navy', 'violet', 'black']

    if method=="tSNE":
        # Apply t-SNE to reduce dimensionality to 2D
        tsne = TSNE(n_components=n_components, random_state=42, 
                    perplexity=30, learning_rate=200, max_iter=1000)
        X_embedded = tsne.fit_transform(X)
    else:
        pca = PCA(n_components=0.9)  # Preserve specific amount of variance
        X_embedded = pca.fit_transform(X).astype('float32')[:,:2]

    # Visualize the reduced 2D data
    plt.figure(figsize=(8, 6))
    n_labels = len(np.unique(y))
    for i in range(n_labels):
        Xi = X_embedded[y==i]
        plt.scatter(Xi[:, 0], Xi[:, 1], c=[colors[i%len(colors)]]*len(Xi), marker='o', 
            s=60, alpha=0.7, label=f'{purpose} {i}')
    fname = f"{root}results/tSNE/{method}_for_{purpose}_{dataset_name}"
    plt.savefig(fname)
    plt.show()

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

datasets = ["Wisconsin"]

method = "tSNE"     # "tSNE" or "PCA"
for dataset_name in datasets:

    set_seeds(seed)

    # get best parameters
    emb_features, n_heads, max_nodes, init_lr, epochs = get_params(dataset_name, root)
    min_lr, decay_steps = 0.0001, 25
    lr_decay = np.power((min_lr/init_lr), decay_steps/epochs)
    heads, sep_learning = n_heads, True
    #heads = 1
    train_frac=0.8
    max_parts = 10

    # load dataset
    data, n_classes = load_dataset(data_folder, dataset_name)
    #data.x = apply_pca(data.x, pca_preserve)
    data.num_features = data.x.shape[1]
    n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
    n_orig_nodes = n_nodes
    print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')

    #sg = get_random_subgraph(data, max_nodes)
    sg = get_subgraphs(dataset_name, data, max_nodes)[0]
    del data
    data_train, data_test = split_data_on_edges(sg, train_frac=train_frac) 
    data_train = tf_GData(data_train)
    data_test = tf_GData(data_test)
    n_nodes, n_edges = data_train.num_nodes, data_train.num_edges    

    affinity_model, aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric = compute_affinty2(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)    
    Z, W = affinity_model.affinity_layer.Z, affinity_model.dense_layer.W
    del affinity_model
    gc.collect()

    
    # plot tSNE of class labels for single head
    for i in range(heads):
        if hasattr(data_train, 'y'):
            if data_train.y is not None:
                purpose = f'labels - head-{i}'
                plot_in_2D(root, method, purpose, dataset_name, Z[i].numpy(), tf.reshape(data_train.y, -1))
    
