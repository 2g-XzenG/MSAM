import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from hparams import *
from moduals import *
from model import *

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


INPUT_DATA_TRAIN, INPUT_DATA_TEST, TARGET_DATA_TRAIN, TARGET_DATA_TEST = load_data(hp.INPUT_DATA, hp.TARGET_DATA)

train_dataset, num_train_batches = get_batch(INPUT_DATA_TRAIN, TARGET_DATA_TRAIN, hp.VOCAB, hp.max_v, hp.max_c, hp.train_batch_size,\
                                             shuffle=True, repeat = True)
test_dataset, num_test_batches = get_batch(INPUT_DATA_TEST, TARGET_DATA_TEST, hp.VOCAB, hp.max_v, hp.max_c, hp.test_batch_size,\
                                           shuffle=False, repeat = False)
iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
input_ccs_seq, input_cost_seq, input_monthcode_seq, input_monthcost_seq, target_ccs_seq, target_cost_seq = iter.get_next()
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)
 
model = MSAM(hp)
train_loss, train_op, code_pred_cost, demo_pred_cost = model.train(input_cost_seq, input_monthcode_seq, input_monthcost_seq, \
                                                             target_ccs_seq, target_cost_seq)
test_loss, topk_ccs, pred_cost, code_pred_cost, demo_pred_cost, eval_summaries = model.eval(input_cost_seq, input_monthcode_seq, input_monthcost_seq, \
                                                                            target_ccs_seq, target_cost_seq)

saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    train_loss_list = []
    for steps in range(total_steps):
        _train_loss, _ = sess.run([train_loss, train_op])
        train_loss_list.append(_train_loss)
        epoch = int(steps / num_train_batches)
        if steps % num_train_batches == 0:
            print("EPOCH: %i, TRAIN-LOSS: %0.2f" %(epoch, np.mean(train_loss_list)))

        if epoch == 3:
            saver.save(sess, "./SAVED/MODEL_"+str(epoch))
            #################################### EVAL ##################################################
            summary_writer = tf.summary.FileWriter(hp.LOGDIR, sess.graph)
            sess.run(test_init_op)
            loss_list = []
            recall_values = []
            predict_cost_list = []
            actual_cost_list = []
            for steps in range(num_test_batches):
                _test_loss, _topk_ccs, _target_ccs, _pred_cost, _target_cost, _eval_summaries = sess.run([test_loss, topk_ccs, target_ccs_seq, pred_cost, target_cost_seq, eval_summaries])
                summary_writer.add_summary(_eval_summaries)
                values = recall_disease(_topk_ccs, [list(np.nonzero(i)[0]) for i in _target_ccs])
                recall_values.extend(values)
                predict_cost_list.extend(_pred_cost)
                actual_cost_list.extend(_target_cost)
                loss_list.append(_test_loss)
            print("N:%i, loss:%0.1f, top-10:%0.2f, MAE:%0.1f"% (len(recall_values), np.mean(loss_list), np.mean(recall_values), mean_absolute_error(predict_cost_list, actual_cost_list)))
            topk_cost(predict_cost_list, actual_cost_list, 100)
            topk_cost(predict_cost_list, actual_cost_list, 500)
            summary_writer.close()






