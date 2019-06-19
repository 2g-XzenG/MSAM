import argparse
  
class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=13234, type=int)
    parser.add_argument('--grouped_vocab_size', default=272, type=int)
    parser.add_argument('--max_v', default=12, type=int)
    parser.add_argument('--max_c', default=100, type=int)

    ## train
    parser.add_argument('--VOCAB', default='../PROCESS/VOCAB', help="VOCAB")
    parser.add_argument('--INPUT_DATA', default='../PROCESS/INPUT_DATA', help="INPUT_DATA")
    parser.add_argument('--TARGET_DATA', default='../PROCESS/TARGET_DATA', help="TARGET_DATA")
    parser.add_argument('--LOGDIR', default="LOG/1", help="log directory")

    # training scheme
    parser.add_argument('--cost_pred_weight', default='../TOOLS/cost_pred_weight', help="cost_pred_weight")
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--ff_dim', default=128, type=int)
    parser.add_argument('--code_dim', default=128, type=int)
    parser.add_argument('--visit_dim', default=128, type=int)
    parser.add_argument('--patient_dim', default=280, type=int) # same as grouped_vocab_size
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--L2_alpha', default=0.001, type=float)
    parser.add_argument('--ce_alpha', default=1, type=float)
    parser.add_argument('--mse_alpha', default=0.0000001, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--eval_every', default=10, type=int)



