# -*- coding:utf-8 -*-
"""
mincheng:mc.cheng@my.cityu.edu.hk
"""
from __future__ import division
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def inputs(option):
    if option == '1L' or option == '2L':
        data_tensor = tf.placeholder(tf.float32, shape=[None, FLAGS.sequence_window, FLAGS.input_dim])
        #data_tensor = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.sequence_window,FLAGS.input_dim])
        label_tensor = tf.placeholder(tf.float32, shape=[None, FLAGS.number_class])
    elif option == 'AL':
        data_tensor = tf.placeholder(tf.float32,shape=[None,FLAGS.scale_levels,FLAGS.sequence_window,FLAGS.input_dim])
        label_tensor = tf.placeholder(tf.float32,shape=[None,FLAGS.number_class])

    elif option == 'HL' or 'HAL' or 'Boost' :
        data_tensor = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.scale_levels,FLAGS.sequence_window,FLAGS.input_dim])
        label_tensor = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.number_class])
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


def loss(predict,label):
    #cost_cross_entropy = -tf.reduce_mean(target * tf.log(prediction))
    cost_cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(predict, label, name=None))  # Sigmoid

    tf.scalar_summary("loss", cost_cross_entropy)
    return cost_cross_entropy

def print_info(tensor,name):
    return tf.Print(tensor, [tensor.get_shape()], "The "+name+" shape is :", first_n=4096, summarize=40)

def inference(data,label,option):
    if option == '1L':#pure one-layer lstm

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)
        val, state = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(label.get_shape()[1])]),name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))

        print("2222")
        print(weight.get_shape())


        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        #tf.scalar_summary("weight", weight)
    elif option == '2L':#two-layer lstm
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)

        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2)
        val, state = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(label.get_shape()[1])]),name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    elif option == 'Boost':#boost lstm (prototype)
        data = tf.transpose(data,[1,0,2,3])
        scale_weight_init_value = 1.0 / FLAGS.scale_levels

        # scale_weight = tf.Variable(tf.constant(scale_weight_init_value,shape=[number_of_scales]), name="scale_weight")
        scale_weight = tf.Variable(tf.constant(scale_weight_init_value, shape=[1]), name="scale_weight")
        training_error = tf.Variable(tf.constant(0., shape=[FLAGS.scale_levels]), name="training_error")
        # training_error = tf.Variable(tf.constant(0.,shape=[]), name="training_error")
        total_error = tf.Variable(tf.constant(0., shape=[]), name="total_error")

        # current_scale = tf.placeholder(tf.int32,shape=None)
        current_scale = tf.Variable(tf.constant(0, shape=[]))

        data_for_lstm = tf.reshape(tf.gather(data, current_scale),
                                   [FLAGS.batch_size, FLAGS.sequence_window, FLAGS.input_dim])
        # data_original_train2 = tf.reshape(data_original_train,(number_of_scales, int(data_original_train.get_shape()[1]) * sequence_window * input_dim))
        print(
        "At first this operation is assigned initial weights to each scale level, and this operation only runs once, after it will update...")

        scale_weight = tf.mul(scale_weight, tf.exp(tf.sub(total_error, tf.gather(training_error, current_scale))))
        # scale_weight = tf.mul(scale_weight,tf.exp(tf.sub(total_error,training_error)))
        print(tf.gather(training_error, 0).get_shape())

        data2 = tf.mul(scale_weight, data_for_lstm)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)
        val, state = tf.nn.dynamic_rnn(lstm_cell, data2, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(label.get_shape()[1])]),name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))

        print(weight.get_shape())


        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    elif option == 'AL':
        u_w_scales_normalized = tf.Variable(tf.constant(1.0 / FLAGS.scale_levels, shape=[1, FLAGS.scale_levels]),
                                            name="u_w")
        u_w_scales_normalized = normalized_scale_levels(u_w_scales_normalized)
        u_w = tf.Variable(tf.random_normal(shape=[1, FLAGS.sequence_window]), name="u_w")

        data = tf.transpose(data,[1,0,2,3])

        # data_original_train = tf.placeholder(tf.float32,[batch_size,sequence_window,input_dim])
        data2 = tf.transpose(data, [1, 2, 3, 0])
        data_merged = batch_vm2(data2, tf.transpose(u_w_scales_normalized))
        data_merged = tf.reshape(data_merged, (-1, FLAGS.sequence_window, FLAGS.input_dim))
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)

        val, state = tf.nn.dynamic_rnn(lstm_cell, data_merged, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(label.get_shape()[1])]),name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)




    elif option == 'HL': #hierarchy lstm
        data_original_train1 = tf.transpose(data, [0, 2, 1, 3])
        data_original_train1 = tf.reshape(data_original_train1,(FLAGS.batch_size*FLAGS.sequence_window,FLAGS.scale_levels,FLAGS.input_dim))

        lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh,state_is_tuple=True)
        val1, state1 = tf.nn.dynamic_rnn(lstm_cell1, data_original_train1, dtype=tf.float32)

        temp1 = tf.transpose(val1, [1, 0, 2])
        last1 = tf.gather(temp1, int(temp1.get_shape()[0]) - 1)
        print(last1.get_shape())
        temp2 = tf.reshape(last1,(FLAGS.batch_size,FLAGS.sequence_window,FLAGS.num_neurons1))


        lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)
        val2, state2 = tf.nn.rnn(lstm_cell2, last1, dtype=tf.float32)



        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(label.get_shape()[1])]),name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))

        val = tf.transpose(temp2, [1, 0, 2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)



        #u_w = tf.Variable(tf.random_normal(shape=[1, FLAGS.scale_levels]), name="u_w")
        #max_pooling_output = tf.nn.max_pool(data_original_train1, [1, FLAGS.sequence_window, 1, 1], \
         #                                   [1, 1, 1, 1], padding='VALID')
        #max_pooling_output_reshape = tf.reshape(max_pooling_output, (FLAGS.batch_size, FLAGS.input_dim, FLAGS.scale_levels))
        #max_pooling_output2 = tf.transpose(max_pooling_output_reshape, [0, 2, 1])


        #Weight_W = tf.Variable(tf.truncated_normal([FLAGS.input_dim, FLAGS.scale_levels]))
        #b_W = tf.Variable(tf.constant(0.1, shape=[1, FLAGS.scale_levels]))
        #batch_W = tf.Variable(tf.constant(0.1, shape=[FLAGS.batch_size, FLAGS.scale_levels]))

        # u_current_levels_temp = tf.matmul(tf.gather(max_pooling_output2,max_pooling_output2.get_shape()[0]-1),Weight_W)+b_W
        #u_current_levels_temp = batch_vm(max_pooling_output2, Weight_W) + b_W
        #print("ddd")
        #print(u_current_levels_temp.get_shape())
        #temp1 = tf.reshape((batch_vm(u_current_levels_temp, tf.transpose(u_w))), (FLAGS.batch_size, FLAGS.scale_levels))
        #temp2 = batch_vm(u_current_levels_temp, tf.transpose(u_w))
        #print(temp2.get_shape())
        #u_current_levels_total = tf.gather(tf.cumsum(tf.exp(tf.transpose(temp1, [1, 0]))), FLAGS.scale_levels - 1)
        #print("eee")
        # u_current_levels_total_list = tf.gather(tf.cumsum(tf.exp(tf.mul(tf.transpose(u_current_levels_temp,[0,1,2]),tf.transpose(u_w)))),number_scale_levels-1)
        # u_current_levels_total_list = tf.cumsum(tf.exp(tf.mul(tf.transpose(u_current_levels_temp,[0,1,2]),tf.transpose(u_w))))
        # print(u_current_levels_total_list.get_shape())
        #u_current_levels = tf.transpose(tf.div(tf.transpose(temp1, [1, 0]), u_current_levels_total), [1, 0])
        #print(u_current_levels.get_shape())

        #out_put_u_w_scale = tf.Print(u_current_levels, [u_current_levels], "The u_current_levels shape is ----------------:",
                                     #first_n=4096, summarize=40)

        #u_w_AAA = tf.Variable(tf.random_normal(shape=[1, FLAGS.sequence_window]), name="u_w_AAA")
        #data_original_train2 = tf.transpose(BGP_Data, [1, 2, 3, 0])
        #print("ccc")
        #print(data_original_train2.get_shape())
        #print(tf.transpose(u_current_levels).get_shape())
        # u_current_levels = tf.reshape(u_current_levels,(batch_size,1,1,number_scale_levels))
        # data_original_train_merged = batch_vm2(data_original_train2,u_current_levels)

        #data_original_train_merged = batch_vm2(data_original_train2, tf.transpose(u_current_levels))
        #data_original_train_merged = tf.transpose(data_original_train_merged, [2, 3, 0, 1])
        #data_original_train_merged = tf.reshape(data_original_train_merged,
         #                                       (FLAGS.batch_size * FLAGS.batch_size, FLAGS.sequence_window, FLAGS.input_dim))

        #index_list = [(i + 1) * (i + 1) - 1 for i in range(FLAGS.batch_size)]
        #data_original_train_merged = tf.gather(data_original_train_merged, index_list)

        # data_original_train_merged = tf.map_fn(batch_vm2,
        #data_original_train_merged = tf.reshape(data_original_train_merged, (FLAGS.batch_size, FLAGS.sequence_window, FLAGS.input_dim))
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.num_neurons1, forget_bias=1.0, activation=tf.nn.tanh)
        #val, state = tf.nn.dynamic_rnn(lstm_cell, data_original_train_merged, dtype=tf.float32)
        #val2 = tf.gather(val, val.get_shape()[0] - 1)
        #Weight_W_AAA = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, FLAGS.sequence_window]))
        #b_W_AAA = tf.Variable(tf.constant(0.1, shape=[FLAGS.sequence_window, FLAGS.sequence_window]))
        #u_current_levels_temp_AAA = tf.matmul(val2, Weight_W_AAA) + b_W_AAA
        #u_current_levels_total_AAA = tf.gather(tf.cumsum(tf.exp(batch_vm(u_current_levels_temp_AAA, tf.transpose(u_w_AAA)))),
         #                                      FLAGS.sequence_window - 1)
        #u_current_levels_AAA = tf.div(tf.exp(batch_vm(u_current_levels_temp_AAA, tf.transpose(u_w_AAA))),
        #                              u_current_levels_total_AAA)

        #target = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.number_class])
        #m_total = batch_vm(tf.transpose(u_current_levels_AAA), val)
        #eight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(target.get_shape()[1])]))
        #bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
        #rediction = tf.nn.softmax(tf.matmul(m_total, weight) + bias)

        """
        out_put_prediction = tf.Print(prediction, [prediction.get_shape()], "The prediction shape is :", first_n=1024,summarize=10)
        output_val = print_info(val, "val")
        output_val = print_info(val2, "val2")
        output_val = print_info(Weight_W_AAA, "Weight_W_AAA")
        output_val = print_info(b_W_AAA, "b_W_AAA")
        output_val = print_info(u_current_levels_temp_AAA, "u_current_levels_temp_AAA")
        output_val = print_info(u_current_levels_total_AAA, "u_current_levels_total_AAA")
        output_val = print_info(u_current_levels_AAA, "u_current_levels_AAA")
        output_val = print_info(m_total, "m_total")
        output_val = print_info(prediction, "prediction")
        """
    elif option == 'HAL':
        data1 = tf.transpose(data,[0,2,3,1])
        u_w = tf.Variable(tf.random_normal(shape=[1, FLAGS.scale_levels]), name="u_w")
        max_pooling_output = tf.nn.max_pool(data1, [1, FLAGS.sequence_window, 1, 1], \
                                            [1, 1, 1, 1], padding='VALID')
        print(max_pooling_output.get_shape())
        print("bbb")
        max_pooling_output_reshape = tf.reshape(max_pooling_output, (FLAGS.batch_size, FLAGS.input_dim, FLAGS.scale_levels))
        max_pooling_output2 = tf.transpose(max_pooling_output_reshape, [0, 2, 1])

        Weight_W = tf.Variable(tf.truncated_normal([FLAGS.input_dim, FLAGS.scale_levels]))
        b_W = tf.Variable(tf.constant(0.1, shape=[1, FLAGS.scale_levels]))
        batch_W = tf.Variable(tf.constant(0.1, shape=[FLAGS.batch_size, FLAGS.scale_levels]))

        # u_current_levels_temp = tf.matmul(tf.gather(max_pooling_output2,max_pooling_output2.get_shape()[0]-1),Weight_W)+b_W
        u_current_levels_temp = batch_vm(max_pooling_output2, Weight_W) + b_W
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

        out_put_u_w_scale = tf.Print(u_current_levels, [u_current_levels],
                                     "The u_current_levels shape is ----------------:", first_n=4096, summarize=40)

        # target = tf.placeholder(tf.float32, [batch_size, number_class])
        # m_total = batch_vm(tf.transpose(u_current_levels),val)
        # u_w_scales_normalized = u_current_levels
        # tf.assign(u_w_scales_normalized,u_current_levels)
        # m_total = tf.mul(tf.transpose(u_current_levels),val)
        # weight = tf.Variable(tf.truncated_normal([num_neurons, int(target.get_shape()[1])]))
        # bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
        # prediction = tf.nn.softmax(tf.matmul(m_total, weight) + bias)

        # u_w_scales_normalized = tf.Variable(tf.constant(1.0/number_scale_levels,shape=[1,number_scale_levels]), name="u_w")
        # u_w_scales_normalized = normalized_scale_levels(u_w_scales_normalized)
        # u_w = tf.Variable(tf.random_normal(shape=[1,sequence_window]), name="u_w")



        # output_data_original_train = tf.Print(data_original_train,[data_original_train],"The Original Train  is :",first_n=4096,summarize=40)

        # data_original_train = tf.placeholder(tf.float32,[batch_size,sequence_window,input_dim])

        u_w_AAA = tf.Variable(tf.random_normal(shape=[1, FLAGS.sequence_window]), name="u_w_AAA")

        data_original_train2 = tf.transpose(data, [1, 2, 3, 0])

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
        Weight_W_AAA = tf.Variable(tf.truncated_normal([FLAGS.num_neurons, FLAGS.sequence_window]))
        b_W_AAA = tf.Variable(tf.constant(0.1, shape=[FLAGS.sequence_window, FLAGS.sequence_window]))
        u_current_levels_temp_AAA = tf.matmul(val2, Weight_W_AAA) + b_W_AAA
        u_current_levels_total_AAA = tf.gather(
            tf.cumsum(tf.exp(batch_vm(u_current_levels_temp_AAA, tf.transpose(u_w_AAA)))), FLAGS.sequence_window - 1)
        u_current_levels_AAA = tf.div(tf.exp(batch_vm(u_current_levels_temp_AAA, tf.transpose(u_w_AAA))),
                                      u_current_levels_total_AAA)
        target = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.number_class])
        m_total = batch_vm(tf.transpose(u_current_levels_AAA), val)
        weight = tf.Variable(tf.truncated_normal([FLAGS.num_neurons1, int(target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
        prediction = tf.nn.softmax(tf.matmul(m_total, weight) + bias)




    return prediction,label

def train(loss):

  _optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
  #_grads = _optimizer.compute_gradients(loss)

  #for var in tf.trainable_variables():
        #tf.histogram_summary(var.op.name, var)

  #for grad, var in _grads:
    #if grad is not None:
      #tf.histogram_summary(var.op.name + '/gradients', grad)

  #_train_op = _optimizer.apply_gradients(_grads, global_step=global_step)
  _train_op = None
  return _optimizer,_train_op






