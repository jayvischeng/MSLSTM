# -*- coding:utf-8 -*-
"""
mincheng:mc.cheng@my.cityu.edu.hk
"""
from __future__ import division
import tensorflow as tf
import printlog
import sys
#from BNLSTM import LSTMCell, BNLSTMCell, orthogonal_initializer
#from tensorflow.nn.rnn_cell import GRUCell
FLAGS = tf.app.flags.FLAGS
def pprint(msg,method=''):
    if not 'Warning' in msg:
        sys.stdout = printlog.PyLogger('',method)
        print(msg)
        sys.stderr.write(msg+'\n')
def inputs(option):
    if 'H' in option:# Hierarcy architecture
        data_tensor = tf.placeholder(tf.float32,shape=[None,FLAGS.scale_levels,FLAGS.sequence_window,FLAGS.input_dim])
        label_tensor = tf.placeholder(tf.float32,shape=[None,FLAGS.number_class])
    else:
        data_tensor = tf.placeholder(tf.float32, shape=[None, FLAGS.sequence_window, FLAGS.input_dim])
        label_tensor = tf.placeholder(tf.float32, shape=[None, FLAGS.number_class])

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
def normalized_scale_levels(scales_list):
    return tf.div(scales_list,tf.gather(tf.gather(tf.cumsum(scales_list, axis=1), 0), int(scales_list.get_shape()[1]-1)))

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
def loss_(predict,label):

    #cost_cross_entropy = -tf.reduce_mean(label * tf.log(predict))
    cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,label))  # Softmax for labels that mutally exclusive
    #cost_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(predict,label))  # Same as above but wihout the need to transfer labels to one-hot vector

    #cost_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predict, label, name=None))  # Sigmoid
    #cost_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(predict, label, name='softmax')

    #--------------------------------------------- Compute cross entropy with various length--------------------------.
    #cross_entropy = label * tf.log(predict)
    #cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=1)
    #cost_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predict, label, name=None))  # Sigmoid
    #mask = tf.sign(tf.reduce_max(tf.abs(label), reduction_indices=1))
    #cross_entropy *= mask
    # Average over actual sequence lengths.
    #cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=0)
    #cross_entropy /= tf.reduce_sum(mask, reduction_indices=0)
    #return tf.reduce_mean(cross_entropy)
    #-----------------------------------------------------------------------------------
    return cost_cross_entropy

def print_info(tensor,name):
    return tf.Print(tensor, [tensor.get_shape()], "The "+name+" shape is :", first_n=4096, summarize=40)

def weight_bias(option):
    if 'H' in option:# Hierarcy architecture
        weights = {
            'in': tf.Variable(tf.random_normal([FLAGS.input_dim, FLAGS.num_neurons1])),
            'out': tf.Variable(tf.random_normal([FLAGS.num_neurons2, FLAGS.number_class]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[FLAGS.num_neurons1, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[FLAGS.number_class, ]))
        }
        if 'A' in option:
            weights['attention'] = tf.Variable(tf.random_normal([FLAGS.num_neurons1, FLAGS.scale_levels]))
            biases['attention'] = tf.Variable(tf.constant(0.1, shape=[FLAGS.scale_levels, ]))
    else:
        weights = {
            'in': tf.Variable(tf.random_normal([FLAGS.input_dim, FLAGS.num_neurons1])),
            'out': tf.Variable(tf.random_normal([FLAGS.num_neurons1, FLAGS.number_class])),
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[FLAGS.num_neurons1, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[FLAGS.number_class, ]))
        }
        if 'A' in option:
            weights['attention'] = tf.Variable(tf.random_normal([FLAGS.num_neurons1, FLAGS.sequence_window]))
            biases['attention'] = tf.Variable(tf.constant(0.1, shape=[FLAGS.sequence_window, ]))

    return weights, biases
def inference(data,label,option,is_training):
    weights,biases = weight_bias(option)
    #X = tf.reshape(data, [-1, FLAGS.input_dim])
    #X_in = tf.matmul(X, weights['in']) + biases['in']#X_in = W*X + b
    #X_in = tf.reshape(X_in, [-1, FLAGS.sequence_window, FLAGS.num_neurons1])#X_in ==> ( batches,  steps,  hidden) 换回3维
    X_in = data
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh,
                                             state_is_tuple=False)
    # lstm_cell = BNLSTMCell(FLAGS.num_neurons1,is_training)# Batch_Normalization
    # lstm_cell = tf.contrib.rnn.BasicRNNCell(FLAGS.num_neurons1, activation=tf.nn.tanh)
    # lstm_cell = tf.contrib.rnn_cell.MultiRNNCell([lstm_cell]*2)
    if option == '1L':#pure one-layer lstm
        val, state = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        #last = tf.unpack(tf.transpose(val, [1, 0, 2]))
    elif option == 'RNN':#pure one-layer RNN

        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.num_neurons1, activation=tf.nn.tanh)
        val, state = tf.nn.dynamic_rnn(rnn_cell, X_in, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

    elif option == '2L':#two-layer lstm
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2,state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1)

    elif option == '3L':#three-layer lstm

        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*3,state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1)

    elif option == 'AL':
        u_ = tf.Variable(tf.random_normal(shape=[1, FLAGS.sequence_window]), name="u_w")
        w_ones = tf.Variable(tf.constant(1.0, shape=[FLAGS.sequence_window,1]),name="u_w_one")
        val, state = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
        val_ = tf.reshape(val,(-1,FLAGS.num_neurons1))
        u_levels = tf.reshape((tf.matmul(val_, weights["attention"]) + biases["attention"]),(-1,FLAGS.sequence_window,FLAGS.sequence_window))
        u_levels_ = tf.transpose(u_levels,[0,1,2])
        u_levels_ = tf.reshape(u_levels_,(FLAGS.sequence_window,-1))
        u_levels_t = tf.exp(tf.matmul(u_,u_levels_))
        w_t = tf.reshape(u_levels_t,(-1,FLAGS.sequence_window))
        w_ = tf.matmul(w_t,w_ones)
        u_w = tf.reshape(tf.div(w_t, w_),(-1,1,FLAGS.sequence_window))
        #u_w = tf.reshape(tf.div(w_t, w_),(-1,FLAGS.sequence_window))
        #val2 = tf.reshape(val,(FLAGS.sequence_window,-1))
        #print(u_w.get_shape())
        #print(val.get_shape())
        last = tf.reshape(tf.matmul(u_w,val),(-1,FLAGS.num_neurons1))
    elif option == 'HL': #hierarchy lstm
        data_train = tf.transpose(X_in, [0, 2, 1, 3])
        data_train = tf.reshape(data_train,(-1,FLAGS.scale_levels,FLAGS.input_dim))

        with tf.variable_scope('1stlayer_hl'):
            lstm_cell_bottom = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)
            val_bottom, state_bottom = tf.nn.dynamic_rnn(lstm_cell_bottom, data_train, dtype=tf.float32)
            temp_b = tf.transpose(val_bottom, [1, 0, 2])
            last_b = tf.gather(temp_b, int(temp_b.get_shape()[0]) - 1)

        temp_ = tf.reshape(last_b,(-1,FLAGS.sequence_window,FLAGS.num_neurons1))

        with tf.variable_scope('2ndlayer_hl'):
            lstm_cell_top = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons2, forget_bias=1.0, activation=tf.nn.tanh)
            val_top, state_top = tf.nn.dynamic_rnn(lstm_cell_top, temp_, dtype=tf.float32)

        val_t = tf.transpose(val_top, [1, 0, 2])
        last = tf.gather(val_t, int(val_t.get_shape()[0]) - 1)

    elif option == 'HAL':
        data_train = tf.transpose(X_in, [0, 2, 1, 3])
        data_train = tf.reshape(data_train, (-1, FLAGS.scale_levels, FLAGS.input_dim))
        with tf.variable_scope('1stlayer_hal'):
            u_w_bottom = tf.Variable(tf.random_normal(shape=[1, FLAGS.scale_levels]), name="u_w_bottom")
            u_w_nor = tf.Variable(tf.constant(1.0, shape=[FLAGS.scale_levels, 1]), name="u_w_nor")
            lstm_cell_bottom = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)
            val_bottom, state_bottom = tf.nn.dynamic_rnn(lstm_cell_bottom, data_train, dtype=tf.float32)
            val_val_bottom_ = tf.reshape(val_bottom, (-1, FLAGS.num_neurons1))

            u_levels_bottom = tf.reshape((tf.matmul(val_val_bottom_, weights["attention"]) + biases["attention"]),
                                               (-1, FLAGS.scale_levels, FLAGS.scale_levels))
            u_levels_t = tf.transpose(u_levels_bottom, [1, 2, 0])
            u_levels_t = tf.reshape(u_levels_t, (FLAGS.scale_levels, -1))
            w_t = tf.reshape(tf.exp(tf.matmul(u_w_bottom, u_levels_t)), (-1, FLAGS.scale_levels))
            w_bottom = tf.matmul(w_t, u_w_nor)
            u_w_bottom_final = tf.reshape(tf.div(w_t, w_bottom), (-1, 1, FLAGS.scale_levels))
            m_temp = tf.matmul(u_w_bottom_final, val_bottom)

        temp = tf.reshape(m_temp, (-1, FLAGS.sequence_window, FLAGS.num_neurons1))
        with tf.variable_scope('2ndlayer_hl'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons2, forget_bias=1.0, activation=tf.nn.tanh)
            val, state = tf.nn.dynamic_rnn(lstm_cell, temp, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)


    output_last = tf.Print(last,[last],message = None,first_n = 4096)
    prediction = tf.matmul(last, weights['out']) + biases['out']
    #prediction = tf.nn.softmax(tf.matmul(last, weights['out']) + biases['out'])

        #try:
     #   return output_u_w,prediction,label
    #except:
    #weights = tf.Variable(tf.constant(1.0, shape=[FLAGS.sequence_window*FLAGS.batch_size, 1,FLAGS.scale_levels]),name="weights")
    #tf.assign(weights,u_w_bottom)
    #return u_w_bottom,prediction,label
    #return prediction,label,output_last
    return prediction,label,last

def train(loss):

  _optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
  _grads = _optimizer.compute_gradients(loss)

  #for var in tf.trainable_variables():
        #tf.histogram_summary(var.op.name, var)

  #for grad, var in _grads:
    #if grad is not None:
      #tf.histogram_summary(var.op.name + '/gradients', grad)

  _train_op = _optimizer.apply_gradients(_grads)
  #_train_op = None
  return _train_op,_optimizer





