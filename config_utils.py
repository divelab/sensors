class SensorConfig(object):
    description= None
   
    update_limit = 3000  # the number of mini-batch before evaluating the model

    # how to encode utterance.
    # bow: add word embedding together
    # rnn: RNN utterance encoder
    # bi_rnn: bi_directional RNN utterance encoder
    sent_type = "bi_rnn"
    sent_cell_size = 300
    att_size = 100
    #sent_cell_size = 200
    #att_size = 100

    position_len = 135

    max_length = 135



    # Network general
    cell_type = "gru"  # gru or lstm
    #embed_size = 150  # word embedding size
    embed_size = 200

    num_layer = 1  # number of context RNN layers

    # Optimization parameters
    op = "adam"
    grad_clip = 15.0  # gradient abs max cut
    init_w = 0.08  # uniform random from [-init_w, init_w]
    batch_size = 5  # mini-batch size
    init_lr = 0.005  # initial learning rate
    lr_hold = 1  # only used by SGD
    lr_decay = 0.6  # only used by SGD
    keep_prob = 0.95  # drop out rate
    improve_threshold = 0.996  # for early stopping
    patient_increase = 2.0  # for early stopping
    early_stop = True
    max_epoch = 30  # max number of epoch of training
    grad_noise = 0.0  # inject gradient noise?







