# main.py

# Set path
root = "D:/Indranil/JRF/Submission/Affinity/codebase2/affNet_LP/affNetR/"
data_dir = "D:/Indranil/ML2/Datasets/recommender/"  
import os
os.chdir(root)

import tensorflow as tf
from load_data import MovieLens1M
from aff_data import AffData
from model import AffNetR
from utils import parse_arg, plot_hist, evaluate 
import numpy as np
import gc

dataset_name, emb_features, num_heads, test_frac, \
    init_lr, lr_decay, dropout, k, epochs = parse_arg()
       
# Load the dataset and train-test split 
dataset_name = 'MovieLens1M' # MovieLens1M, Amazon_Book, Gowalla, Yelp2018
dataset = MovieLens1M(data_dir=data_dir, neg_ratio=1)
has_features = dataset.has_features
aff_data = AffData(dataset)
if dataset_name not in ['MovieLens1M']: # subgraph if too large
    aff_data = aff_data.induced_subgraph_on_top_items()
aff_data.summary()
train_data, test_data = aff_data.split(test_frac=test_frac)
print("\nTrain data staistics:")
train_data.count_isolated_nodes()
train_data.user_item_edge_counts()
print()
del dataset, aff_data
gc.collect()

results_dict = {'precision':[], 'recall':[], 'ndcg':[]}
for i in range(10):
    # create AffNetR model and optimizer
    model = AffNetR(train_data.num_users, train_data.num_items, 
                    train_data.num_features, emb_features, 
                    num_heads, dropout, has_features=has_features)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init_lr,
        decay_steps=5,
        decay_rate=lr_decay,  # after 1000 steps, lr = 0.96 * previous
        staircase=True     # if True, it decays in discrete intervals
    )
    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_hist, test_hist, prec_list, rec_list, x_norm_list, z_norm_list, beta_list = model.train(train_data, test_data, epochs, optim)
    if i==0:
        plot_hist(train_hist, test_hist, "train loss", "test loss")
        plot_hist(prec_list, rec_list, "precision", "recall", "precision vs recall")
    
    # evaluate
    test_results = evaluate(model, test_data, k=k)
    results_dict['precision'].append(test_results['precision'])
    results_dict['recall'].append(test_results['recall'])
    results_dict['ndcg'].append(test_results['ndcg'])

    del model
    gc.collect()

precision = np.mean(results_dict['precision'])
recall = np.mean(results_dict['recall'])
ndcg = np.mean(results_dict['ndcg'])

print(f"precision: {precision:.4f}, recall: {recall:.4f}, ndcg: {ndcg:.4f}")


# wrap up
del train_data, test_data
gc.collect()

