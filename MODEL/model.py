import tensorflow as tf
from moduals import *

class MSAM():
    def __init__(self, hp):
        self.hp = hp

    def build_gragh(self, input_cost, input_monthcode, input_monthcost, \
                    target_ccs, target_cost, training):

        with tf.variable_scope("GRAPH", reuse=tf.AUTO_REUSE):
            code_embed_mat = get_token_embeddings("code_embed_mat", self.hp.vocab_size, self.hp.code_dim)
            codemask_embed_mat = tf.get_variable("code_mask_matrix",initializer=[[0.]]+[[1.]]*(self.hp.vocab_size-1),trainable=False)
            cost_pred_weight = tf.get_variable("cost_pred_weight", initializer=np.load(self.hp.cost_pred_weight, allow_pickle=True),trainable=True)
        
        
            code_enc = tf.nn.embedding_lookup(code_embed_mat, input_monthcode) #self.batch_size, self.max_v, self.max_c
            code_enc *= self.hp.code_dim**0.5
            code_enc = tf.layers.dropout(code_enc, self.hp.dropout_rate, training=training)
            code_enc = tf.reshape(code_enc, shape = (-1, self.hp.max_c, self.hp.code_dim))

            for i in range(self.hp.num_blocks):
                with tf.variable_scope("code_level_num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    code_enc = multihead_attention(scope_name = "code_level", 
                                                   queries=code_enc,
                                                   keys=code_enc,
                                                   values=code_enc,
                                                   num_heads=self.hp.num_heads,
                                                   dropout_rate=self.hp.dropout_rate,
                                                   training=training)
                    code_enc = ff("code_level", code_enc, [self.hp.ff_dim, self.hp.code_dim])
            code_enc = tf.reshape(code_enc, shape = (-1, self.hp.max_v, self.hp.max_c, self.hp.code_dim))
            code_mask = tf.nn.embedding_lookup(codemask_embed_mat, input_monthcode)
            code_enc = tf.multiply(code_enc, code_mask)

            visit_enc = tf.reduce_sum(code_enc, axis=2)
            visit_enc += positional_encoding(visit_enc, maxlen=self.hp.max_v)
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("visit_level_num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    visit_enc = multihead_attention(scope_name = "visit_level",
                                                    queries=visit_enc,
                                                    keys=visit_enc,
                                                    values=visit_enc,
                                                    num_heads=self.hp.num_heads,
                                                    dropout_rate=self.hp.dropout_rate,
                                                    training=training)
                    visit_enc = ff("visit_level", visit_enc, num_units=[self.hp.ff_dim, self.hp.visit_dim])
            patient_enc = tf.reduce_sum(visit_enc, axis = 1)
            patient_enc = tf.layers.dropout(patient_enc, rate=self.hp.dropout_rate, training=training)   
            patient_enc = tf.layers.dense(patient_enc, self.hp.grouped_vocab_size, activation=None)
            
            code_pred_cost = tf.nn.relu(tf.matmul(tf.nn.sigmoid(patient_enc), cost_pred_weight))
            demo_pred_cost = tf.layers.dense(tf.layers.dense(tf.reshape(input_monthcost,(-1,12)), 10), 1, activation=tf.nn.relu)
            
            return patient_enc, cost_pred_weight, code_pred_cost, demo_pred_cost

    def train(self, input_cost, input_monthcode, input_monthcost, \
                target_ccs, target_cost):
        patient_enc, cost_pred_weight, code_pred_cost, demo_pred_cost = self.build_gragh(input_cost, input_monthcode, input_monthcost, \
                                                                       target_ccs, target_cost, training=True)
        pred_cost = code_pred_cost + demo_pred_cost

        var = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if len(v.shape) >1])
        pred_loss = tf.losses.mean_squared_error(labels = tf.reshape(target_cost,(-1,1)), predictions = pred_cost)
        aux_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = target_ccs, logits = patient_enc)
        loss = pred_loss * self.hp.mse_alpha + aux_loss * self.hp.ce_alpha 
        train_op = tf.train.AdamOptimizer().minimize(loss)
        
        return loss, train_op, code_pred_cost, demo_pred_cost


    def eval(self, input_cost, input_monthcode, input_monthcost, \
            target_ccs, target_cost):

        patient_enc, cost_pred_weight, code_pred_cost, demo_pred_cost = self.build_gragh(input_cost, input_monthcode, input_monthcost, \
                                                                       target_ccs, target_cost, training=False)         
        pred_cost = code_pred_cost + demo_pred_cost
        topk_ccs = tf.nn.top_k(patient_enc, k=10)

        var = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if len(v.shape) >1])
        pred_loss = tf.losses.mean_squared_error(labels = tf.reshape(target_cost,(-1,1)), predictions = pred_cost)
        aux_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = target_ccs, logits = patient_enc)
        loss = pred_loss * self.hp.mse_alpha + aux_loss * self.hp.ce_alpha 

        summaries = tf.summary.merge_all()

        return loss, topk_ccs[1], pred_cost, code_pred_cost, demo_pred_cost, summaries











