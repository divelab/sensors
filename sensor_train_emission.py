import os
import time
import csv

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils import SensorConfig as Config
from models.srnn import SensorRNN
from datautils import Data_helper
from datautils import evaluate

# constants
tf.app.flags.DEFINE_string("work_dir", "./work_dir", "Experiment results directory.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("task", "emission", "Prediction task")
tf.app.flags.DEFINE_string("test_path", "run1582061407", "the dir to load checkpoint for forward only")
tf.app.flags.DEFINE_bool("test", False, "Predict test results")
FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    # config for training
    config = Config()
    config.batch_size = 1

    # config for validation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 1

    # configuration for testing
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1
    test_config.max_length = 135

    pp(config)

    best_test = np.inf

    # get data set
    train_feed = Data_helper(FLAGS.task+'_input.txt', FLAGS.task+'_output.txt', config.batch_size, config.position_len)
    test_feed = Data_helper(FLAGS.task+'_input_test.txt', FLAGS.task+'_output_test.txt', test_config.batch_size, config.position_len)


    if FLAGS.resume:
        log_dir = os.path.join(FLAGS.work_dir, FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "run"+str(int(time.time())))

    # begin training
    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "model"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = SensorRNN(sess, config, None, log_dir=log_dir, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            test_model = SensorRNN(sess, test_config, None, log_dir=None, forward=True, scope=scope)

        # write config to a file for logging
        if not FLAGS.resume:
            with open(os.path.join(log_dir, "run.log"), "wb") as f:
                f.write(pp(config, output=False).encode())

        # create a folder by force
        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        print("Created models with fresh parameters.")
        sess.run(tf.global_variables_initializer())

        if FLAGS.resume:
            print(("Reading dm models parameters from %s" % FLAGS.test_path))
            model_checkpoint_path = FLAGS.test_path
            model.saver.restore(sess, model_checkpoint_path)

        if FLAGS.test:
            test_label, test_prediction, test_loss, weights = test_model.test(sess, test_feed)
            evaluate(test_feed.label, test_prediction)
            print(test_loss)

            with open(FLAGS.test_path+'.csv', mode='w') as file:
                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(test_label)):
                    file_writer.writerow([test_label[i], test_prediction[i]])

        else:
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")

            global_t = 1

            for epoch in range(config.max_epoch):
                print((">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval())))
                global_t, loss = model.train(global_t, sess, train_feed)
                test_sensors, test_prediction, test_loss, weights = test_model.test(sess, test_feed)
                print(("Epoch ",epoch+1 , " average loss is ", loss, " test loss is ",test_loss))
                #if test_loss < best_test:
                print("Save model!!")
                model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
                best_test = test_loss


if __name__ == "__main__":
    main()













