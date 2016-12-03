# -*- coding:utf-8 -*-
"""
mincheng:mc.cheng@my.cityu.edu.hk
"""
from __future__ import division
import tensorflow as tf
import mslstm
flags = tf.app.flags
flags.DEFINE_string('data_dir',"~/Dropbox/MSLSTM/data","""Directory for storing data set""")
flags.DEFINE_string('is_use_m_scale',False,"""Run with multi-scale or not""")
flags.DEFINE_string('input_dim','33',"""Input dimension size""")
flags.DEFINE_string('num_neurons1','200',"""Number of hidden units""")
flags.DEFINE_string('num_neurons2','200',"""Number of hidden units""")
flags.DEFINE_string('sequence_window','20',"""Sequence window size""")
flags.DEFINE_string('scale_levels','10',"""Scale level value""")
flags.DEFINE_string('number_class','2',"""Number of output nodes""")
flags.DEFINE_string('wave_type','db1',"""Type of wavelet""")
flags.DEFINE_string('batch_size','200',"""Batch size""")
flags.DEFINE_string('max_steps','10000',"""Number of batches to run""")
flags.DEFINE_string('learning_rate','0.002',"""Learning rate""")
flags.DEFINE_string('log_dir','./log/',"""Directory where to write the event logs""")

FLAGS = flags.FLAGS

def main(unused_argv):
    FLAGS.is_use_m_scale = False
    print(FLAGS.data_dir)
    print(FLAGS.is_use_m_scale)

if __name__ == "__main__":
    tf.app.run()

def sess_init():
    init = tf.initialize_all_tables()
    config = tf.ConfigProto()
    config.gpu_option.allow_groth = True
    sess = tf.Session(config=config)
    sess.run(init)
    return sess

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0,name="global_step",trainable=False)
        data_x,data_y = mslstm.inputs()
