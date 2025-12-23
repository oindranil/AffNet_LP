import tensorflow as tf
from tensorflow.keras import layers

def rmse_loss(edge_index, affinity):

    indices = edge_index.indices
    true_vals = edge_index.values
    pred_vals = tf.gather_nd(affinity, indices)
    
    pred_vals = tf.clip_by_value(pred_vals, 0.0, 1.0)
    true_vals = tf.cast(true_vals, tf.float32)

    mse = tf.reduce_mean(tf.square(pred_vals - true_vals))
    return tf.sqrt(mse)

def soft_f1_loss(edge_index, affinity, epsilon=1e-7):
    indices = edge_index.indices
    true_vals = edge_index.values
    pred_vals = tf.gather_nd(affinity, indices)
    
    """
    pred_vals = tf.clip_by_value(pred_vals, 0.0, 1.0)
    true_vals = tf.cast(true_vals, tf.float32)

    tp = tf.reduce_sum(pred_vals * true_vals)
    fp = tf.reduce_sum(pred_vals * (1.0 - true_vals))
    fn = tf.reduce_sum((1.0 - pred_vals) * true_vals)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    """
    diff = true_vals - pred_vals
    mismatches = tf.abs(diff)
    matches = 1 - mismatches
    tp = tf.reduce_sum(tf.math.multiply(matches, pred_vals))
    fp = tf.reduce_sum(tf.math.multiply(mismatches, pred_vals))
    fn = tf.reduce_sum(tf.math.multiply(mismatches, 1 - pred_vals))    

    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    loss = 1 - f1    
    
    return (loss, precision, recall) # Loss to minimize

class EmbeddingBlock(tf.keras.layers.Layer):
    def __init__(self, emb_features, dropout):
        super().__init__()
        self.dense1 = layers.Dense(emb_features)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout)

        self.dense2 = layers.Dense(emb_features)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout)

        self.activation = layers.LeakyReLU()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.dropout2(x, training=training)

        return x

class AffinityLayer(tf.keras.layers.Layer):
    def __init__(self, num_users, emb_features, num_heads):
        super(AffinityLayer, self).__init__()

    def call(self, X_emb, Z):
        affinity = tf.linalg.matmul(Z, tf.transpose(X_emb)) 
        Z_norm = tf.linalg.norm(Z, axis=-1)
        X_norm = tf.linalg.norm(X_emb, axis=-1)
        affinity = tf.divide(affinity, X_norm + 0.000001)
        affinity = tf.transpose(tf.divide(tf.transpose(affinity, perm=[0,2,1]), 
                      tf.expand_dims(Z_norm + 0.000001, axis=1)), perm=[0,2,1])
        return(affinity)

# Define a model that uses the custom layers
class AffNetR(tf.keras.Model):
    def __init__(self, num_users, num_items, num_features, emb_features, 
                 num_heads, dropout, has_features=False):
        super(AffNetR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.emb_features = emb_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.has_features = has_features

        # trainable parameters 
        #self.W = self.add_weight(name='W',
        #    shape=[self.num_features, self.emb_features],
        #    initializer=tf.random_normal_initializer())
        if not self.has_features:
            self.X = self.add_weight(name='X',
                shape=[self.num_items, self.emb_features],
                initializer=tf.random_normal_initializer())
        self.Z = self.add_weight(name='Z',
            shape=[self.num_heads, self.num_users, self.emb_features],
            initializer=tf.random_normal_initializer())
       	self.beta = self.add_weight(
       	    name="beta", shape=(),
       	    initializer=tf.keras.initializers.Constant(1.0),
       	    trainable=True,
       	    constraint=tf.keras.constraints.NonNeg()
       	)
        # Add the layers
        self.embedding_layer = EmbeddingBlock(self.emb_features, self.dropout)
        self.affinity_layer = AffinityLayer(self.num_users, self.emb_features, self.num_heads)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)

    def forward(self, x, training=False):
        #X_emb = tf.linalg.matmul(X,self.W) # embedding layer
        #X_emb = self.dropout1(X_emb, training=training)
        if x is None:
            x = self.X
        else:
            x = self.embedding_layer(x, training=True) # embedding layer
        aff = self.affinity_layer(x, self.Z)
        aff = tf.reduce_max(aff, axis=0) # pooling layer
        aff = (aff+1) / 2.0 # scaling layer
        aff = tf.math.pow(aff, self.beta) # exponential layer
        return(aff)

    def backward(self, edge_index, aff):   
        loss, precision, recall = soft_f1_loss(edge_index, aff)
        return(loss, precision, recall)

    def train(self, train_data, test_data, epochs, optim):
        train_hist, test_hist = [], []
        prec_list, rec_list = [], []
        x_norm_list, z_norm_list, beta_list = [], [], []
        print("    ", end='')
        for e in range(epochs):
            print(f"\rTraining AffNetR ... {int(100*e/epochs):>3}%", end="", flush=True)
            with tf.GradientTape() as tape:
                affinity = self.forward(train_data.x, training=True)
                train_loss, precision, recall = self.backward(train_data.edge_index, affinity)
            grads = tape.gradient(train_loss, self.trainable_weights)
            optim.apply_gradients(zip(grads, self.trainable_weights))     
            test_loss, _, _ = soft_f1_loss(test_data.edge_index, affinity)
            train_hist.append(train_loss.numpy())
            test_hist.append(test_loss.numpy())
            prec_list.append(precision.numpy())
            rec_list.append(recall.numpy())
            if not self.has_features:
                x_norm = tf.norm(self.X)
            else:
                x_norm = tf.norm(train_data.x)
            z_norm = tf.norm(self.Z)
            x_norm_list.append(x_norm)
            z_norm_list.append(z_norm)
            beta_list.append(self.beta.numpy())
        print("\b\b\b\b... done. ")
        return(train_hist, test_hist, prec_list, rec_list, x_norm_list, z_norm_list, beta_list)

