# -*- coding:utf-8 -*-
"""
mincheng:mc.cheng@my.cityu.edu.hk
"""
from __future__ import division
import os
import tensorflow as tf
import mslstm
import loaddata
flags = tf.app.flags
flags.DEFINE_string('data_dir',os.path.join(os.getcwd(),'data'),"""Directory for storing data set""")
flags.DEFINE_string('is_multi_scale',False,"""Run with multi-scale or not""")
flags.DEFINE_string('input_dim',33,"""Input dimension size""")
flags.DEFINE_string('num_neurons1',200,"""Number of hidden units""")
flags.DEFINE_string('num_neurons2',200,"""Number of hidden units""")
flags.DEFINE_string('sequence_window',20,"""Sequence window size""")
flags.DEFINE_string('scale_levels',10,"""Scale level value""")
flags.DEFINE_string('number_class',2,"""Number of output nodes""")
flags.DEFINE_string('wave_type','db1',"""Type of wavelet""")
flags.DEFINE_string('pooling_type','max pooling',"""Type of wavelet""")
flags.DEFINE_string('batch_size',200,"""Batch size""")
flags.DEFINE_string('max_epochs',20,"""Number of epochs to run""")
flags.DEFINE_string('learning_rate',0.002,"""Learning rate""")
flags.DEFINE_string('is_add_noise',False,"""Whether add noise""")
flags.DEFINE_string('noise_ratio',0,"""Noise ratio""")
flags.DEFINE_string('log_dir','./log/',"""Directory where to write the event logs""")

FLAGS = flags.FLAGS





def sess_run(commander,data,label):
    global sess, data_x, data_y
    return sess.run(commander, {data_x: data, data_y: label})


def train(filename,tab_cv,cross_cv):
    global sess, data_x, data_y

    x_train, y_train, x_test, y_test = loaddata.GetData(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, 'Attention', FLAGS.data_dir,
                                                        filename, FLAGS.sequence_window, tab_cv, cross_cv,
                                                        Multi_Scale=FLAGS.is_multi_scale, Wave_Let_Scale=FLAGS.scale_levels,
                                                        Wave_Type=FLAGS.wave_type)
    with tf.Graph().as_default():
        tf.set_random_seed(1337)

        #global_step = tf.Variable(0,name="global_step",trainable=False)
        data_x,data_y = mslstm.inputs()

        prediction, label = mslstm.inference(data_x,data_y)
        loss = mslstm.loss(prediction, label)
        optimizer,train_op = mslstm.train(loss)
        minimize = optimizer.minimize(loss)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        summary_op = tf.merge_all_summaries()


        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)


        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
        saver = tf.train.Saver()

        epoch_training_loss_list = []
        epoch_training_acc_list = []
        epoch_val_loss_list = []
        epoch_val_acc_list = []
        weight_list = []
        early_stopping = 10
        no_of_batches = int(len(x_train) / FLAGS.batch_size)

        for i in range(FLAGS.max_epochs):
            if early_stopping > 0:
                pass
            else:
                epoch_stop = i + 1
                break
            ptr = 0
            for j in range(no_of_batches):
                inp, out = x_train[ptr:ptr + FLAGS.batch_size], y_train[ptr:ptr + FLAGS.batch_size]
                inp2, out2 = x_test[ptr:ptr + FLAGS.batch_size], y_test[ptr:ptr + FLAGS.batch_size]

                #inp, out = x_train[:, ptr:ptr + FLAGS.batch_size], y_train[ptr:ptr + FLAGS.batch_size]
                #inp2, out2 = x_test[:, ptr:ptr + FLAGS.batch_size], y_test[ptr:ptr + FLAGS.batch_size]

                ptr += FLAGS.batch_size
                sess_run(minimize,inp,out)
                training_acc, training_loss = sess_run((accuracy,loss),inp,out)
                val_acc, val_loss = sess_run((accuracy,loss),inp2,out2)

                if j%5 == 0:
                    summary_str = sess.run(summary_op, {data_x: inp, data_y: out})
                    summary_writer.add_summary(summary_str, i * no_of_batches)

                epoch_val_loss_list.append(val_loss)
                epoch_val_acc_list.append(val_acc)



            print("Epoch %s" % (str(i + 1)) + ">" * 20 + "=" + "train_accuracy: %s, train_loss: %s" % (str(training_acc), str(training_loss)) \
                  + ",\tval_accuracy: %s, val_loss: %s" % (str(val_acc), str(val_loss)))




            try:
                max_val_acc = epoch_val_acc_list[-2]
            except:
                max_val_acc = 0

            if epoch_val_acc_list[-1] < max_val_acc:
                early_stopping -= 1
            elif epoch_val_acc_list[-1] >= max_val_acc:
                early_stopping = 10
        #incorrect = sess.run(error, {data: x_test, data_y: y_test})
        """
        try:
            weight_list = []
            result = []1
            ptr = 0
            for t in range(int(len(y_test) / batch_size)):

                inp_test, out_test = x_test[:, ptr:ptr + batch_size], y_test[ptr:ptr + batch_size]
                if len(inp_test[0]) != batch_size: continue

                result.append(sess.run(prediction, {data_x: inp_test, data_y: out_test}))
                weight_list.append(sess.run(out_put_u_w_scale, {data_x: inp_test, data_y: out_test}))

                ptr += batch_size
            x_ = [i for i in range(number_scale_levels)]
            y_ = [i for i in range(int(len(y_test) / batch_size))]
            x_, y_ = np.meshgrid(x_, y_)
            plt.plot(np.array(x_), np.array(y_), np.array(result))
            plt.show()
        except:
            pass
        """
def main(unused_argv):

    FLAGS.is_use_m_scale = False
    print(FLAGS.data_dir)
    print(FLAGS.is_use_m_scale)
    train("HB_AS_Leak.txt",0,2)
if __name__ == "__main__":
    tf.app.run()
