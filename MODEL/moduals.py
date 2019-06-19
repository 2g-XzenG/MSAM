import tensorflow as tf
import _pickle as pickle
import numpy as np
import math
from sklearn.model_selection import train_test_split

######################################## LOADING DATA ############################################

def data_generator(input_ccs_seq, input_cost_seq, input_monthcode_seq, input_monthcost_seq, \
                   target_ccs_seq, target_cost_seq):
    for x, y, z, l, m, n in zip(input_ccs_seq, input_cost_seq, input_monthcode_seq, input_monthcost_seq, \
                             target_ccs_seq, target_cost_seq):
        yield x, y, z, l, m, n

def padding_visit_level(data, shapes):
    pad_data = np.zeros(shapes)
    for i, p in enumerate(data):
        for j, v in enumerate(p):
            for k, c in enumerate(v):
                pad_data[i][j][k] = c
    return pad_data

def convert_int(data, vocab2idx):
    converted_data = [[[vocab2idx[c] for c in v] for v in p] for p in data]
    return converted_data

def process_mcode(data, vocab2idx, max_v, max_c):
    input_mcode_int = convert_int(data, vocab2idx)
    shapes = (len(input_mcode_int), max_v, max_c)
    pad_input_mcode = padding_visit_level(input_mcode_int, shapes)
    return pad_input_mcode

def process_ccs(data, vocab2idx):
    shapes = (len(data), len(vocab2idx))
    pad_data = np.zeros(shapes)
    for i, p in enumerate(data):
        for j, c in enumerate(p):
            pad_data[i][vocab2idx[c]-1] = 1  # zero index -> not padding!
    return pad_data

def get_batch(INPUT_DATA, TARGET_DATA, vocab_fpath,\
              max_v, max_c, batch_size, shuffle=True, repeat=True):
    '''
    input shape:  1. input code seq;  (N, max_V (12), max_C (128))
    output shape: 1. output cost seq; (N,)
    '''

    input_ccs_seq, input_cost_seq, input_monthcode_seq, input_monthcost_seq = INPUT_DATA
    target_ccs_seq, target_cost_seq, _, _ = TARGET_DATA
    code2idx, idx2code, ccs2idx, idx2ccs = pickle.load(open(vocab_fpath,"rb"))

    input_monthcode_seq = process_mcode(input_monthcode_seq, code2idx, max_v, max_c)
    target_ccs_seq = process_ccs(target_ccs_seq, ccs2idx)
    input_ccs_seq = process_ccs(input_ccs_seq, ccs2idx)

    shapes = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None,]), \
              tf.TensorShape([None]), tf.TensorShape([]))
    types = (tf.int32, tf.float64, tf.int32, tf.float32, \
             tf.int32, tf.float32)
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_shapes=shapes,
                                             output_types=types,
                                             args=(input_ccs_seq, input_cost_seq, input_monthcode_seq, input_monthcost_seq, \
                                                   target_ccs_seq, target_cost_seq))

    if shuffle: dataset = dataset.shuffle(1234)
    if repeat: dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    num_batches = math.ceil(len(input_monthcode_seq)/batch_size)
    return dataset, num_batches

def load_data(INPUT_DATA_fpath, TARGET_DATA_fpath):
    input_ccs_seq, input_cost_seq, input_monthcode_seq, input_monthcost_seq = pickle.load(open(INPUT_DATA_fpath,"rb"))
    target_ccs_seq, target_cost_seq, target_monthcode_seq, target_monthcost_seq = pickle.load(open(TARGET_DATA_fpath,"rb"))

    input_ccs_seq_train, input_ccs_seq_test,\
    input_cost_seq_train, input_cost_seq_test,\
    input_monthcode_seq_train, input_monthcode_seq_test,\
    input_monthcost_seq_train, input_monthcost_seq_test = train_test_split(input_ccs_seq, input_cost_seq,\
                                                                           input_monthcode_seq, input_monthcost_seq,\
                                                                           test_size=0.20, random_state=1234)

    target_ccs_seq_train, target_ccs_seq_test,\
    target_cost_seq_train, target_cost_seq_test,\
    target_monthcode_seq_train, target_monthcode_seq_test,\
    target_monthcost_seq_train, target_monthcost_seq_test = train_test_split(target_ccs_seq, target_cost_seq,\
                                                                             target_monthcode_seq, target_monthcost_seq,\
                                                                             test_size=0.20, random_state=1234)

    return [input_ccs_seq_train, input_cost_seq_train, input_monthcode_seq_train, input_monthcost_seq_train],\
           [input_ccs_seq_test, input_cost_seq_test, input_monthcode_seq_test, input_monthcost_seq_test],\
           [target_ccs_seq_train, target_cost_seq_train, target_monthcode_seq_train, target_monthcost_seq_train],\
           [target_ccs_seq_test, target_cost_seq_test, target_monthcode_seq_test, target_monthcost_seq_test],\

######################################## BUILDING MODEL ############################################

def get_token_embeddings(name, vocab_size, num_units, zero_pad=True):
    embeddings = tf.get_variable(name, dtype=tf.float32, shape=(vocab_size, num_units))
    if zero_pad:embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
    return embeddings

def mask(inputs, queries=None, keys=None, type=None):
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        masks = tf.expand_dims(masks, 1)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  
    elif type in ("q", "query", "queries"):
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  
        masks = tf.expand_dims(masks, -1)  
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]]) 
        outputs = inputs*masks
    return outputs

def scaled_dot_product_attention(scope_name, Q, K, V, dropout_rate, training):
    with tf.variable_scope(scope_name+ "dotproduct_attention", reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs /= d_k ** 0.5
        outputs = mask(outputs, Q, K, type="key")
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
        outputs = mask(outputs, Q, K, type="query")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V)  
    return outputs

def ln(scope_name, inputs, epsilon = 1e-8):
    with tf.variable_scope(scope_name+"_ln", reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs-mean)/((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs

def multihead_attention(scope_name, queries, keys, values, num_heads, dropout_rate, training):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope_name + "_multihead_attention", reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model, use_bias=False)
        K = tf.layers.dense(keys, d_model, use_bias=False) 
        V = tf.layers.dense(values, d_model, use_bias=False)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) 

        outputs = scaled_dot_product_attention(scope_name, Q_, K_, V_, dropout_rate, training)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries
        outputs = ln(scope_name, outputs)
    return outputs

def ff(scope_name, inputs, num_units):
    with tf.variable_scope(scope_name+"_feedforward", reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1])
        outputs += inputs        
        outputs = ln(scope_name, outputs)
    return outputs

def positional_encoding(inputs, maxlen, masking=True):
    E = inputs.get_shape().as_list()[-1]
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope("positional_encoding", reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) 
        position_enc = np.array([[pos / np.power(10000, (i-i%2)/E) for i in range(E)] for pos in range(maxlen)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) 
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        if masking: outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        return tf.to_float(outputs)


######################################## EVALUATE MODEL ############################################

def recall_disease(topk_list, true_list):
    values = []
    for i, j in zip(topk_list, true_list):
        if len(j)==0:continue
        values.append(len(set(i).intersection(set(j)))/len(j))
    return values

def topk_cost(pred_list, true_list, k):
    pred_list = np.array(pred_list).reshape(-1,)
    true_list = np.array(true_list).reshape(-1,)
    topk_pred_index = pred_list.argsort()[-k:][::-1]
    topk_true_index = true_list.argsort()[-k:][::-1]
    topk_pred_cost = pred_list[topk_pred_index]
    topk_true_cost = true_list[topk_pred_index]
    print("pred-topk:%i, true-topk:%i, acutal-topk:%i"%(np.mean(topk_pred_cost), np.mean(topk_true_cost), np.mean(sorted(true_list)[-k:])))
    print("selected top-k people:", len(set(topk_pred_index).intersection(topk_true_index)))










