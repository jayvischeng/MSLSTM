# -*- coding:utf-8 -*-
"""
mincheng:mc.cheng@my.cityu.edu.hk
"""
from __future__ import division
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def inputs(option):
    #data_tensor = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.sequence_window,FLAGS.input_dim,FLAGS.scale_levels])
    if option > 0:
        data_tensor = tf.placeholder(tf.float32,shape=[FLAGS.scale_levels,FLAGS.batch_size,FLAGS.sequence_window,FLAGS.input_dim])
        label_tensor = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.number_class])
    else:
        data_tensor = tf.placeholder(tf.float32,shape=[None,FLAGS.sequence_window,FLAGS.input_dim])
        label_tensor = tf.placeholder(tf.float32,shape=[None,FLAGS.number_class])

    return data_tensor,label_tensor

def batch_vm(v, m):
  shape = tf.shape(v)
  rank = shape.get_shape()[0].value
  v = tf.expand_dims(v, rank)

  vm = tf.mul(v, m)

  return tf.reduce_sum(vm, rank-1)

def batch_vm2(x, m):
  [input_size, output_size] = m.get_shape().as_list()

  input_shape = tf.shape(x)
  batch_rank = input_shape.get_shape()[0].value - 1
  batch_shape = input_shape[:batch_rank]
  output_shape = tf.concat(0, [batch_shape, [output_size]])

  x = tf.reshape(x, [-1, input_size])
  y = tf.matmul(x, m)

  y = tf.reshape(y, output_shape)

  return y

def loss(predict,label):
    #cost_cross_entropy = -tf.reduce_mean(target * tf.log(prediction))
    cost_cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(predict, label, name=None))  # Sigmoid
    return cost_cross_entropy

def print_info(tensor,name):
    return tf.Print(tensor, [tensor.get_shape()], "The "+name+" shape is :", first_n=4096, summarize=40)

def inference(data,label,option=0):
    if option > 0:
        data_original_train1 = tf.transpose(data, [1, 2, 3, 0])

        u_w = tf.Variable(tf.random_normal(shape=[1, FLAGS.scale_levels]), name="u_w")
        max_pooling_output = tf.nn.max_pool(data_original_train1, [1, FLAGS.sequence_window, 1, 1], \
                                            [1, 1, 1, 1], padding='VALID')
        max_pooling_output_reshape = tf.reshape(max_pooling_output, (FLAGS.batch_size, FLAGS.input_dim, FLAGS.scale_levels))
        max_pooling_output2 = tf.transpose(max_pooling_output_reshape, [0, 2, 1])


        Weight_W = tf.Variable(tf.truncated_normal([FLAGS.input_dim, FLAGS.scale_levels]))
        b_W = tf.Variable(tf.constant(0.1, shape=[1, FLAGS.scale_levels]))
        batch_W = tf.Variable(tf.constant(0.1, shape=[FLAGS.batch_size, FLAGS.scale_levels]))

        # u_current_levels_temp = tf.matmul(tf.gather(max_pooling_output2,max_pooling_output2.get_shape()[0]-1),Weight_W)+b_W
        u_current_levels_temp = batch_vm(max_pooling_output2, Weight_W) + b_W
        print("ddd")
        print(u_current_levels_temp.get_shape())
        temp1 = tf.reshape((batch_vm(u_current_levels_temp, tf.transpose(u_w))), (FLAGS.batch_size, FLAGS.scale_levels))
        temp2 = batch_vm(u_current_levels_temp, tf.transpose(u_w))
        print(temp2.get_shape())
        u_current_levels_total = tf.gather(tf.cumsum(tf.exp(tf.transpose(temp1, [1, 0]))), FLAGS.scale_levels - 1)
        print("eee")
        # u_current_levels_total_list = tf.gather(tf.cumsum(tf.exp(tf.mul(tf.transpose(u_current_levels_temp,[0,1,2]),tf.transpose(u_w)))),number_scale_levels-1)
        # u_current_levels_total_list = tf.cumsum(tf.exp(tf.mul(tf.transpose(u_current_levels_temp,[0,1,2]),tf.transpose(u_w))))
        # print(u_current_levels_total_list.get_shape())
        u_current_levels = tf.transpose(tf.div(tf.transpose(temp1, [1, 0]), u_current_levels_total), [1, 0])
        print(u_current_levels.get_shape())

        #out_put_u_w_scale = tf.Print(u_current_levels, [u_current_levels], "The u_current_levels shape is ----------------:",
                                     #first_n=4096, summarize=40)

        u_w_AAA = tf.Variable(tf.random_normal(shape=[1, FLAGS.sequence_window]), name="u_w_AAA")
        data_original_train2 = tf.transpose(data, [1, 2, 3, 0])
        #print("ccc")
        #print(data_original_train2.get_shape())
        #print(tf.transpose(u_current_levels).get_shape())
        # u_current_levels = tf.reshape(u_current_levels,(batch_size,1,1,number_scale_levels))
        # data_original_train_merged = batch_vm2(data_original_train2,u_current_levels)

        data_original_train_merged = batch_vm2(data_original_train2, tf.transpose(u_current_levels))
        data_original_train_merged = tf.transpose(data_original_train_merged, [2, 3, 0, 1])
        data_original_train_merged = tf.reshape(data_original_train_merged,
                                                (FLAGS.batch_size * FLAGS.batch_size, FLAGS.sequence_window, FLAGS.input_dim))

        index_list = [(i + 1) * (i + 1) - 1 for i in range(FLAGS.batch_size)]
        data_original_train_merged = tf.gather(data_original_train_merged, index_list)

        # data_original_train_merged = tf.map_fn(batch_vm2,
        data_original_train_merged = tf.reshape(data_original_train_merged, (FLAGS.batch_size, FLAGS.sequence_window, FLAGS.input_dim))
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)
        val, state = tf.nn.dynamic_rnn(lstm_cell, data_original_train_merged, dtype=tf.float32)
        val2 = tf.gather(val, val.get_shape()[0] - 1)
        Weight_W_AAA = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, FLAGS.sequence_window]))
        b_W_AAA = tf.Variable(tf.constant(0.1, shape=[FLAGS.sequence_window, FLAGS.sequence_window]))
        u_current_levels_temp_AAA = tf.matmul(val2, Weight_W_AAA) + b_W_AAA
        u_current_levels_total_AAA = tf.gather(tf.cumsum(tf.exp(batch_vm(u_current_levels_temp_AAA, tf.transpose(u_w_AAA)))),
                                               FLAGS.sequence_window - 1)
        u_current_levels_AAA = tf.div(tf.exp(batch_vm(u_current_levels_temp_AAA, tf.transpose(u_w_AAA))),
                                      u_current_levels_total_AAA)

        target = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.number_class])
        m_total = batch_vm(tf.transpose(u_current_levels_AAA), val)
        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
        prediction = tf.nn.softmax(tf.matmul(m_total, weight) + bias)

        out_put_prediction = tf.Print(prediction, [prediction.get_shape()], "The prediction shape is :", first_n=1024,
                                      summarize=10)
        output_val = print_info(val, "val")
        output_val = print_info(val2, "val2")
        output_val = print_info(Weight_W_AAA, "Weight_W_AAA")
        output_val = print_info(b_W_AAA, "b_W_AAA")
        output_val = print_info(u_current_levels_temp_AAA, "u_current_levels_temp_AAA")
        output_val = print_info(u_current_levels_total_AAA, "u_current_levels_total_AAA")
        output_val = print_info(u_current_levels_AAA, "u_current_levels_AAA")
        output_val = print_info(m_total, "m_total")
        output_val = print_info(prediction, "prediction")
    else:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
        val, state = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons, int(label.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
def train():
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)





