#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import re
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
from tensorflow.python.ops import variable_scope

from . import decoder_fn_lib
from . import utils
from models.seq2seq import dynamic_rnn_decoder
from .utils import gaussian_kld
from .utils import get_bi_rnn_encode
from .utils import get_bow
from .utils import get_rnn_encode
from .utils import norm_log_liklihood
from .utils import sample_gaussian


class BaseTFModel(object):
    global_t = tf.placeholder(dtype=tf.int32, name="global_t")
    learning_rate = None
    scope = None

    @staticmethod
    def print_model_stats(tvars):
        total_parameters = 0
        for variable in tvars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    @staticmethod
    def get_rnncell(cell_type, cell_size, keep_prob, num_layer):
        # thanks for this solution from @dimeldo
        cells = []
        for _ in range(num_layer):
            if cell_type == "gru":
                cell = rnn_cell.GRUCell(cell_size)
            else:
                cell = rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)

            if keep_prob < 1.0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            cells.append(cell)

        if num_layer > 1:
            cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        else:
            cell = cells[0]

        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def train(self, global_t, sess, train_feed):
        raise NotImplementedError("Train function needs to be implemented")

    def valid(self, *args, **kwargs):
        raise NotImplementedError("Valid function needs to be implemented")

    def batch_2_feed(self, *args, **kwargs):
        raise NotImplementedError("Implement how to unpack the back")

    def optimize(self, sess, config, loss, log_dir):
        if log_dir is None:
            return
        # optimization
        if self.scope is None:
            tvars = tf.trainable_variables()
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        grads = tf.gradients(loss, tvars)
        if config.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip))
        # add gradient noise
        if config.grad_noise > 0:
            grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
            grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]

        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(config.init_lr)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
        self.print_model_stats(tvars)
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)


class SensorRNN(BaseTFModel):

    def __init__(self, sess, config, api, log_dir, forward, scope=None):
        self.vocab_size = 32
        self.sess = sess
        self.scope = scope
        self.sent_cell_size = config.sent_cell_size
        self.max_length = config.max_length


        with tf.name_scope("io"):
            # all dialog context and known attributes
            self.sensor = tf.placeholder(dtype=tf.int32, shape=(None, None), name="sensor")

            # target response given the dialog context
            self.output = tf.placeholder(dtype=tf.float32, shape=(None,), name="output")

            # optimization related variables
            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")




        with variable_scope.variable_scope("wordEmbedding"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, config.embed_size], dtype=tf.float32)
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            embedding = self.embedding * embedding_mask

            input_embedding = embedding_ops.embedding_lookup(embedding, self.sensor)

            length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(input_embedding), reduction_indices=2)),reduction_indices=1)
            length_mask = tf.to_int32(length_mask)
            mask = tf.sequence_mask(length_mask, self.max_length, tf.float32)

            one = tf.ones_like(mask)
            bias = one - mask
            bias = -100000 * bias

            if config.sent_type == "bow":
                pass
                # input_embedding, sent_size = get_bow(input_embedding)
   

            elif config.sent_type == "rnn":
                pass
                # sent_cell = self.get_rnncell("gru", self.sent_cell_size, config.keep_prob, 1)
                # input_embedding, sent_size = get_rnn_encode(input_embedding, sent_cell, scope="sent_rnn")

            elif config.sent_type == "bi_rnn":
                fwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                bwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                input_embedding, sent_size, hidden = get_bi_rnn_encode(input_embedding, fwd_sent_cell, bwd_sent_cell, scope="sent_bi_rnn")
                input_embedding = tf.expand_dims(input_embedding, 1)
                query = tf.get_variable("query", [config.att_size], dtype=tf.float32)
                #input_embedding = layers.fully_connected(input_embedding, config.att_size, activation_fn=None, biases_initializer=None, scope="att")
                hidden_project = layers.fully_connected(hidden, config.att_size, activation_fn=None, biases_initializer=None, scope="att")
                vector_attn = tf.reduce_sum(tf.multiply(hidden_project, query), axis=2, keep_dims=True)
                bias = tf.expand_dims(bias, -1)
                attention_weights = tf.nn.softmax(vector_attn+bias, dim=1)
                self.weights = attention_weights
                attention = hidden*attention_weights
                feature = tf.reduce_sum(attention, 1)
        
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            # reshape input into dialogs

            if config.keep_prob < 1.0:
                feature = tf.nn.dropout(feature, config.keep_prob)

            # convert floors into 1 hot
            predict = layers.fully_connected(feature, 1, activation_fn=None, scope="fc")
            self.predict = tf.squeeze(predict)
            self.loss = tf.losses.absolute_difference(self.output, self.predict)

            tf.summary.scalar("loss", self.loss)

            self.summary_op = tf.summary.merge_all()
            self.optimize(sess, config, self.loss, log_dir)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)



    def train(self, global_t, sess, train_feed, update_limit=5000):

        local_t = 0
        losses = []
        start_time = time.time()
        epoch_train = train_feed.batch_iter()
        for data, label in epoch_train:
            feed_dict = { self.sensor: data, self.output: label}
            _, sum_op, loss = sess.run([self.train_ops, self.summary_op, self.loss], feed_dict)
            self.train_summary_writer.add_summary(sum_op, global_t)
            losses.append(loss)

            global_t += 1
            local_t += 1

        # finish epoch!
        epoch_time = time.time() - start_time
        avg_loss = np.mean(losses)

        return global_t, avg_loss

    def valid(self, name, sess, valid_feed):
        pass

    def test(self, sess, test_feed):
        losses = []
        prediction = []
        outputs = []
        weights = []
        epoch_test = test_feed.batch_iter(False)
        for data, label in epoch_test:
            feed_dict = { self.sensor: data, self.output: label}
            output, predict, loss, weight = sess.run([self.output, self.predict , self.loss, self.weights], feed_dict)
            outputs.append(output)
            losses.append(loss)
            prediction.append(predict)
            weights.append(weight)

        # finish epoch!
        avg_loss = np.mean(losses)

        return outputs, prediction, avg_loss, weights
