import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# parse command line arguments
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MovieLens1M')
    parser.add_argument('--emb_features', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--test_frac', type=float, default=0.3)
    parser.add_argument('--init_lr', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()
    
    dataset_name = args.dataset
    emb_features = args.emb_features
    num_heads = args.num_heads
    test_frac = args.test_frac
    init_lr = args.init_lr
    lr_decay = args.lr_decay
    dropout = args.dropout
    k = args.k       
    epochs = args.epochs

    return(dataset_name, emb_features, num_heads, test_frac, 
           init_lr, lr_decay, dropout, k, epochs)

def plot_hist(train_hist, test_hist, label1, label2, title="Training vs Test Loss", save_path=None):
    assert len(train_hist) == len(test_hist), "Mismatch in number of epochs"

    epochs = range(1, len(train_hist) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_hist, label=label1, color='blue', linewidth=2)
    plt.plot(epochs, test_hist, label=label2, color='orange', linewidth=2, linestyle='--')

    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    #plt.ylabel('Loss', fontsize=14)
    plt.xlim([0, len(epochs)])
#    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()

def evaluate(model, test_data, k=10):
    """
    Evaluate model on test_data, only over masked items (pos + neg edges).
    """
    pred_affinity = model.forward(test_data.x, training=False)  # shape: [num_users, num_items]
    num_items = pred_affinity.shape[1]
    if k == 'all' or k is None:
        k = num_items

    # Convert SparseTensor to COO
    indices = test_data.edge_index.indices.numpy()  # shape: [num_edges, 2]
    values = test_data.edge_index.values.numpy()    # shape: [num_edges]

    user_item_dict = {}
    for idx, (u, i) in enumerate(indices):
        user_item_dict.setdefault(u, []).append((i, values[idx]))

    precisions, recalls, ndcgs = [], [], []

    for u, items_labels in user_item_dict.items():
        item_indices = [i for i, _ in items_labels]
        labels = np.array([l for _, l in items_labels])
        scores = pred_affinity[u].numpy()[item_indices]

        if np.sum(labels) == 0:
            continue

        # Ranking
        top_k = np.argsort(scores)[::-1][:k]
        rel_at_k = labels[top_k]

        precision = np.sum(rel_at_k) / k
        recall = np.sum(rel_at_k) / np.sum(labels)

        def dcg(rels):
            return np.sum((2**rels - 1) / np.log2(np.arange(2, 2 + len(rels))))

        dcg_k = dcg(rel_at_k)
        idcg_k = dcg(np.sort(labels)[::-1][:k])
        ndcg = dcg_k / idcg_k if idcg_k > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    #suffix = f"@{k}"
    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "ndcg": np.mean(ndcgs),
    }

def accuracy(model, data):
    # Get predicted affinity scores (num_users x num_items)
    pred_affinity = model.forward(data.x, training=False)

    # Ground truth sparse matrix
    true_affinity = data.edge_index  # tf.SparseTensor with ratings in [0,1]

    # Convert SparseTensor to dense binary relevance matrix
    true_affinity = tf.sparse.reorder(true_affinity)
    true_dense = tf.sparse.to_dense(true_affinity)
    binary_true = tf.where(true_dense > 0, 1.0, 0.0).numpy()  # shape: (U, I)

    # Convert predictions to NumPy
    pred_affinity_np = pred_affinity.numpy()

    # Threshold predictions at 0.5
    binary_pred = (pred_affinity_np >= 0.5).astype(np.float32)
    
    # Flatten both arrays to match shape: (num_users * num_items,)
    y_true = binary_true.flatten()
    y_pred = binary_pred.flatten()
    
    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    return(acc)

def metrics(model, data):
    affinity = model.forward(data.x, training=False)
    indices = data.edge_index.indices
    true_vals = data.edge_index.values.numpy()
    pred_vals = tf.gather_nd(affinity, indices)
    pred_vals = tf.cast(pred_vals > 0.5, tf.float32).numpy()

    diff = true_vals - pred_vals
    mismatches = tf.abs(diff)
    matches = 1 - mismatches
    tp = tf.reduce_sum(tf.math.multiply(matches, pred_vals))
    fp = tf.reduce_sum(tf.math.multiply(mismatches, pred_vals))
    fn = tf.reduce_sum(tf.math.multiply(mismatches, 1 - pred_vals))    

    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    print(tp.numpy(), fp.numpy(), fn.numpy())
    return(precision, recall)
