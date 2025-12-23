import tensorflow as tf
from utils import eval_link_pred

# given true and predicted values of binary tensor
# compute TP, FP, TN, FN & F1-score
def F_loss(y_true, y_pred):
    y_diff = y_true- y_pred
    mismatches = tf.abs(y_diff)
    matches = 1 - mismatches
    tp = tf.reduce_sum(tf.math.multiply(matches, y_pred))
    fp = tf.reduce_sum(tf.math.multiply(mismatches, y_pred))
    fn = tf.reduce_sum(tf.math.multiply(mismatches, 1 - y_pred))    

    precision = tp / (tp+fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    loss = 1 - f1    
    return(loss)   

# decides which entries in affinity matrix to be considered in loss
def get_loss_mask(pos_adj, neg_adj):
    n_pos = tf.reduce_sum(pos_adj)
    n_neg = tf.reduce_sum(neg_adj)  
    n_total = n_pos+n_neg
    n_pos, n_neg = n_pos/n_total, n_neg/n_total
    n_nodes = pos_adj.shape[0]
    inverted_eye = tf.ones((n_nodes, n_nodes))-tf.eye(n_nodes) 
    #no_edge_mask = tf.cast(tf.convert_to_tensor(no_edge_mask), tf.float32)
    loss_mask = tf.math.multiply((n_neg*pos_adj+n_pos*neg_adj), inverted_eye)
    return(loss_mask)

# Custom pooling Layer: ignores cells outside loss-mask
class PoolingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PoolingLayer, self).__init__()

    def call(self, inputs):
        # Apply pooling on the inputs
        pooled = tf.reduce_max(inputs, axis=0)
        return(pooled)

# Custom Scaling Layer: scales input to the range [0, 1]
class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ScalingLayer, self).__init__()

    def call(self, inputs):
        scaled = (inputs + 1)/2 # convert to (0,1)
        return(scaled)

# Custom Exponentiation Layer: raises inputs to learnable beta
class ExponentiationLayer(tf.keras.layers.Layer):
    def __init__(self, sep_learning):
        super(ExponentiationLayer, self).__init__()
        # Define a learnable parameter for exponentiation
       	self.beta = self.add_weight(
       	    name="beta", shape=(),
       	    initializer=tf.keras.initializers.Constant(1.0),
       	    trainable=True,
       	    constraint=tf.keras.constraints.NonNeg()
       	)

    def call(self, inputs):
        # Apply beta-exponentiation
        x = tf.math.pow(inputs, self.beta) # apply beta
        
        return(x)

# Custom masking Layer: ignores cells outside loss-mask
class MaskingLayer(tf.keras.layers.Layer):
    def __init__(self, n_nodes):
        super(MaskingLayer, self).__init__()
        self.loss_mask = tf.ones((n_nodes, n_nodes))

    def call(self, inputs):
        # Apply masking on the inputs
        masked = tf.math.multiply(inputs, self.loss_mask)        
        return(masked)

# Embedding Layer
class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, in_nodes, out_nodes):
        super(DenseLayer, self).__init__()
        self.W = self.add_weight(name='W',
            shape=[in_nodes, out_nodes],
            initializer=tf.random_normal_initializer())

    def call(self, X):
        # Apply embedding on the X
        X_emb = tf.linalg.matmul(X, self.W)
        return(X_emb)

class AffinityLayer(tf.keras.layers.Layer):
    def __init__(self, n_nodes, emb_features, n_heads):
        super(AffinityLayer, self).__init__()
        self.Z = self.add_weight(name='Z',
            shape=[n_heads, n_nodes, emb_features],
            initializer=tf.random_normal_initializer())

    def call(self, X):        
        affinity = tf.linalg.matmul(self.Z, tf.transpose(X)) 
        Z_norm = tf.linalg.norm(self.Z, axis=-1)
        X_norm = tf.linalg.norm(X, axis=-1)
        affinity = tf.divide(affinity, X_norm + 0.000001)
        affinity = tf.transpose(tf.divide(tf.transpose(affinity, perm=[0,2,1]), 
                      tf.expand_dims(Z_norm + 0.000001, axis=1)), perm=[0,2,1])
        return(affinity)

# Define a model that uses the custom layers
class AffNet(tf.keras.Model):
    def __init__(self, dataset_name, n_nodes, n_features, is_directed, 
                 emb_features, n_heads, sep_learning):
        super(AffNet, self).__init__()

        self.dataset_name = dataset_name
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.emb_features = emb_features
        self.n_heads = n_heads
        self.is_directed = is_directed
        self.sep_learning = sep_learning

        # Add the layers
        self.dense_layer = DenseLayer(n_features, emb_features)
        self.affinity_layer = AffinityLayer(n_nodes, emb_features, n_heads)
        self.scaling_layer = ScalingLayer()
        self.exponentiation_layer = ExponentiationLayer(sep_learning)
        self.pooling_layer = PoolingLayer()
        self.masking_layer = MaskingLayer(n_nodes)

    def forward(self, X):
        # Pass the inputs through the custom layers
        X_emb = self.dense_layer(X)
        aff = self.affinity_layer(X_emb)
        aff = self.pooling_layer(aff)
        if not self.is_directed:
            aff_tranposed = tf.transpose(aff)
            aff = (aff + aff_tranposed)/2.0        
        aff = self.scaling_layer(aff)
        aff = self.exponentiation_layer(aff)
        return(aff)

    def backward(self, adj, aff):
        adj_masked = self.masking_layer(adj)
        aff_masked = self.masking_layer(aff)

        loss = F_loss(adj_masked, aff_masked)
        return(loss)

    def train(self, data_train, data_test, epochs, optim):
        pos_edge_index, neg_edge_index = data_train.pos_edge_index, data_train.neg_edge_index
        pos_adj = tf.sparse.to_dense(pos_edge_index)
        neg_adj = tf.sparse.to_dense(neg_edge_index)
        self.masking_layer.loss_mask = get_loss_mask(pos_adj, neg_adj)
        hist_loss, hist_aff, hist_metric = [], [], []
        print("    ", end='')
        for e in range(epochs):
            print(f"\rTraining affinity model ... {int(100*e/epochs):>3}%", end="", flush=True)
            #print(f"\b\b\b\b{int(100*e/epochs):>3}%", end='')
            with tf.GradientTape() as tape:
                affinity = self.forward(data_train.x)
                loss = self.backward(pos_adj, affinity)
                #metric_name, metric_value = eval_link_pred(self.dataset_name, affinity, data_test)
                metric_name, metric_value = None, None
            grads = tape.gradient(loss, self.trainable_weights)
            optim.apply_gradients(zip(grads, self.trainable_weights))     
            aff_score = self.get_affinity_score(data_train.x)
            hist_loss.append(loss.numpy())
            hist_aff.append(aff_score)
            hist_metric.append(metric_value)
        print("\b\b\b\b... done. ")
        return(hist_loss, hist_aff, hist_metric)

    def get_affinity_matrix(self, X):
        affinity = self.forward(X)
        return(affinity)

    def get_affinity_score(self, X): 
        affinity = self.get_affinity_matrix(X)
        diag_sum = tf.reduce_sum(tf.linalg.diag_part(affinity)).numpy()
        diag_mean = diag_sum/self.n_nodes
        return(diag_mean)

def compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs):
    # create & train affinity model
    affinity_model = AffNet(dataset_name, data_train.num_nodes, data_train.num_features, 
                is_directed, emb_features, heads, sep_learning)        
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
    hist_loss, hist_aff, hist_metric = affinity_model.train(data_train, data_test, epochs, optim)
    
    # get affinity matrix for plotting later
    aff_matrix = affinity_model.get_affinity_matrix(data_train.x)

    # compute affinity score
    aff_h = affinity_model.get_affinity_score(data_train.x)
    beta = affinity_model.exponentiation_layer.beta.numpy() 

    del affinity_model
    return(aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric)

# this fn does exactly same thing except that it returns the model
def compute_affinty2(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs):
    # create & train affinity model
    affinity_model = AffNet(dataset_name, data_train.num_nodes, data_train.num_features, 
                is_directed, emb_features, heads, sep_learning)        
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
    hist_loss, hist_aff, hist_metric = affinity_model.train(data_train, data_test, epochs, optim)
    
    # get affinity matrix for plotting later
    aff_matrix = affinity_model.get_affinity_matrix(data_train.x)

    # compute affinity score
    aff_h = affinity_model.get_affinity_score(data_train.x)
    beta = affinity_model.exponentiation_layer.beta.numpy() 

    return(affinity_model, aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric)

