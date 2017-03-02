# -*- coding:utf-8 -*-
"""
mincheng:mc.cheng@my.cityu.edu.hk
"""
from __future__ import division
import sys
import printlog
import datetime
import os
import time
from baselines import sclearn
import evaluation
from collections import defaultdict
import tensorflow as tf
import mslstm
import loaddata
import numpy as np
import visualize
from sklearn.metrics import accuracy_score
import ucr_load_data
from baselines import nnkeras,sclearn
import matplotlib.pyplot as plt
flags = tf.app.flags
flags.DEFINE_string('data_dir',os.path.join(os.getcwd(),'BGP_Data'),"""Directory for storing BGP_Data set""")
flags.DEFINE_string('is_multi_scale',False,"""Run with multi-scale or not""")
flags.DEFINE_string('input_dim',33,"""Input dimension size""")
flags.DEFINE_string('num_neurons1',120,"""Number of hidden units""")#HAL(hn1=32,hn2=16)
flags.DEFINE_string('num_neurons2',16,"""Number of hidden units""")
flags.DEFINE_string('sequence_window',23,"""Sequence window size""")
flags.DEFINE_string('attention_size',10,"""attention size""")
flags.DEFINE_string('scale_levels',10,"""Scale level value""")
flags.DEFINE_string('number_class',2,"""Number of output nodes""")
flags.DEFINE_string('max_grad_norm',5,"""Maximum gradient norm during training""")
flags.DEFINE_string('wave_type','haar',"""Type of wavelet""")
flags.DEFINE_string('pooling_type','max pooling',"""Type of wavelet""")
flags.DEFINE_string('batch_size',1000,"""Batch size""")
flags.DEFINE_string('max_epochs',200,"""Number of epochs to run""")
flags.DEFINE_string('learning_rate',0.01,"""Learning rate""")
flags.DEFINE_string('is_add_noise',False,"""Whether add noise""")
flags.DEFINE_string('noise_ratio',0,"""Noise ratio""")
flags.DEFINE_string('option','AL',"""Operation[1L:one-layer lstm;2L:two layer-lstm;HL:hierarchy lstm;HAL:hierarchy attention lstm]""")
flags.DEFINE_string('log_dir','./log/',"""Directory where to write the event logs""")
flags.DEFINE_string('output','./output/',"""Directory where to write the results""")

FLAGS = flags.FLAGS


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]

    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpbt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
def pprint(msg,method=''):
    #if not 'Warning' in msg:
    if 1<0:
        sys.stdout = printlog.PyLogger('',method+'_'+str(FLAGS.num_neurons1))
        print(msg)
        try:
            sys.stderr.write(msg+'\n')
        except:
            pass
        #sys.stdout.flush()
#def sess_run(commander,data,label):
    #global sess, data_x, data_y
    #return sess.run(commander, {data_x: data, data_y: label})

def train_lstm(method,filename,cross_cv,tab_cross_cv,result_list_dict,evaluation_list):
    global tempstdout
    FLAGS.option = method
    dropout = 0.8

    #x_train, y_train, x_test, y_test = loaddata.GetData(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, 'Attention', FLAGS.data_dir,
    #                                                    filename, FLAGS.sequence_window, tab_cross_cv, cross_cv,
    #                                                   Multi_Scale=FLAGS.is_multi_scale, Wave_Let_Scale=FLAGS.scale_levels,
    #                                                    Wave_Type=FLAGS.wave_type)

    x_train, y_train, x_test, y_test = ucr_load_data.load_ucr_data(FLAGS.is_multi_scale,filename)
    loaddata.Multi_Scale_Plotting_2(x_train)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    if FLAGS.is_multi_scale:
        FLAGS.scale_levels = x_train.shape[1]
        FLAGS.sequence_window = x_train.shape[len(x_train.shape) - 2]
        FLAGS.input_dim = x_train.shape[-1]
        FLAGS.number_class = y_train.shape[1]
        FLAGS.batch_size = int(y_train.shape[0])
    else:
        FLAGS.sequence_window = x_train.shape[1]
        FLAGS.input_dim = x_train.shape[-1]
        FLAGS.number_class = y_train.shape[1]
        FLAGS.batch_size = 100
    #g = tf.Graph()
    #with g.as_default():
    with tf.variable_scope("middle")as scope:
        tf.set_random_seed(1337)
        #global_step = tf.Variable(0,name="global_step",trainable=False)
        data_x,data_y = mslstm.inputs(FLAGS.option)
        #output_u_w,prediction, label = mslstm.inference(data_x,data_y,FLAGS.option)

        is_training = tf.placeholder(tf.bool)


        prediction, label = mslstm.inference(data_x,data_y,FLAGS.option,is_training)
        loss = mslstm.loss_(prediction, label)
        optimizer = mslstm.train(loss)
        minimize = optimizer.minimize(loss)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #summary_op = tf.merge_all_summaries()
        weights = tf.Variable(tf.constant(0.321, shape=[len(y_test)*FLAGS.sequence_window, 1, FLAGS.scale_levels]),
                              name="weights123")

        saver = tf.train.Saver({"my_weights": weights})
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
    #init_op = tf.initialize_all_variables()
    #with tf.Session() as sess:
        sess.run(init_op)
        #summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
        #saver = tf.train.Saver()
        epoch_training_loss_list = []
        epoch_training_acc_list = []
        epoch_val_loss_list = []
        epoch_val_acc_list = []
        early_stopping = 100
        no_of_batches = int(len(x_train) / FLAGS.batch_size)
        #visualize.Quxian_Plotting(x_train, y_train, 0, "Train_"+str(tab_cross_cv)+'_'+FLAGS.option)
        #visualize.Quxian_Plotting(x_test, y_test, 2, "Test_"+str(tab_cross_cv)+'_'+FLAGS.option)
        for i in range(FLAGS.max_epochs):
            #if early_stopping > 0:
                #pass
            #else:
                #break
            for j_batch in iterate_minibatches(x_train,y_train,FLAGS.batch_size,shuffle=True):
                inp, out = j_batch
                sess.run(minimize, {data_x: inp, data_y: out, is_training:True})
                training_acc, training_loss = sess.run((accuracy, loss), {data_x: inp, data_y: out,is_training:True})
                #sys.stdout = tempstdout

                val_acc, val_loss = sess.run((accuracy, loss), {data_x:x_test, data_y:y_test,is_training:True})
            pprint(
                FLAGS.option + "_Epoch%s" % (str(i + 1)) + ">" * 3 + str(FLAGS.wave_type) + '-' + str(FLAGS.scale_levels) + '-' + str(FLAGS.learning_rate)+'-'+str(FLAGS.num_neurons1)+'-'+str(FLAGS.num_neurons2)+ ">>>=" + "train_accuracy: %s, train_loss: %s" % (
                str(training_acc), str(training_loss)) \
                + ",\tval_accuracy: %s, val_loss: %s" % (str(val_acc), str(val_loss)), method + '_' + filename)




            epoch_training_loss_list.append(training_loss)
            epoch_training_acc_list.append(training_acc)
            epoch_val_loss_list.append(val_loss)
            epoch_val_acc_list.append(val_acc)


            try:
                max_val_acc = epoch_val_acc_list[-2]
            except:
                max_val_acc = 0

            if epoch_val_acc_list[-1] < max_val_acc:
                early_stopping -= 1
            elif epoch_val_acc_list[-1] >= max_val_acc:
                early_stopping = 10
        try:
            weights_results = sess.run(output_u_w, {data_x:x_test, data_y: y_test})
            sess.run(weights.assign(weights_results))
        except:
            pass

        #weights = output_u_w.eval(session=sess)
        #weights = saver.restore(sess, "./tf_tmp/model.ckpt")
        #pprint(weights)
        #pprint("ggg")
        #weight_list = return_max_index(weights)
        result = sess.run(prediction, {data_x:x_test, data_y: y_test})

    saver.save(sess, "./tf_tmp/model.ckpt")
    sess.close()
    #results = evaluation.evaluation(y_test, result)#Computing ACCURACY, F1-Score, .., etc
    sys.stdout = tempstdout

    y_test2 = np.array(evaluation.ReverseEncoder(y_test))
    result2 = np.array(evaluation.ReverseEncoder(result))
    results = accuracy_score(y_test2, result2)
    #print(y_test2)
    #print(result2)
    #print(results)
    with open(os.path.join(os.path.join(os.getcwd(),'stat'),"StatFalseAlarm_" + filename + "_True.txt"), "w") as fout:
        for tab in range(len(y_test2)):
            fout.write(str(int(y_test2[tab])) + '\n')
    with open(os.path.join(os.path.join(os.getcwd(),'stat'),"StatFalseAlarm_" + filename + "_" + method + "_" + "_Predict.txt"), "w") as fout:
        for tab in range(len(result2)):
            fout.write(str(int(result2[tab])) + '\n')

    for each_eval, each_result in results.items():
        result_list_dict[each_eval].append(each_result)


    with open(os.path.join(FLAGS.output, "TensorFlow_Log" + filename + ".txt"), "a")as fout:
        if not FLAGS.is_multi_scale:
            outfileline = FLAGS.option + "_____epoch:" + str(FLAGS.max_epochs) + ",_____learning rate:" + str(FLAGS.learning_rate) + ",_____multi_scale:" + str(FLAGS.is_multi_scale) + "hidden_nodes: "+str(FLAGS.num_neurons1)+"/"+str(FLAGS.num_neurons2) + "\n"
        else:
            outfileline = FLAGS.option + "_____epoch:" + str(FLAGS.max_epochs) + ",____wavelet:"+str(FLAGS.wave_type) + ",_____learning rate:" + str(FLAGS.learning_rate) + ",_____multi_scale:" + str(FLAGS.is_multi_scale) + ",_____train_set_using_level:" + str(FLAGS.scale_levels) + "hidden_nodes: "+str(FLAGS.num_neurons1)+"/"+str(FLAGS.num_neurons2) + "\n"

        fout.write(outfileline)
        for eachk, eachv in result_list_dict.items():
            fout.write(eachk + ": " + str(round(np.mean(eachv), 3)) + ",\t")
        fout.write('\n')

    return epoch_training_acc_list,epoch_val_acc_list,epoch_training_loss_list,epoch_val_loss_list

def train_classic(method,filename,cross_cv,tab_cross_cv,result_list_dict,evaluation_list):
    return sclearn.Basemodel(method,filename,cross_cv,tab_cross_cv)
def train(method,filename,cross_cv,tab_cross_cv,wave_type='db1'):
    global data_x, data_y
    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY", "F1_SCORE", "AUC", "G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []
    for tab_cv in range(cross_cv):
        if tab_cv == tab_cross_cv: continue
        if 'L' in method:
            sys.stdout = tempstdout
            if method == '1L' or method == '2L':
                FLAGS.learning_rate = 0.02
                FLAGS.is_multi_scale = False
            elif 'AL' == method:
                FLAGS.learning_rate = 0.02
                FLAGS.is_multi_scale = False
            else:
                FLAGS.learning_rate = 0.05
                FLAGS.is_multi_scale = True
                FLAGS.wave_type = wave_type
            return train_lstm(method,filename,cross_cv,tab_cross_cv,result_list_dict,evaluation_list)
        else:
            sys.stdout = tempstdout
            return train_classic(method,filename,cross_cv,tab_cross_cv,result_list_dict,evaluation_list)


def main(unused_argv):
    global tempstdout

    #main function
    #filename_list = ["HB_AS_Leak.txt"]
    filename_list = ["Two_Patterns"]

    #wave_type_list =['db1','db2','haar','coif1','db1','db2','haar','coif1','db1','db2']
    wave_type_list = ['haar']

    multi_scale_value_list = [2,3,4,5,6,10]

    case_label = {'1L':'LSTM','2L':'2-LSTM','AL':'ALSTM','HL':'HLSTM','HAL':'HALSTM'}
    #case = ['1L','2L','AL','HL','HAL']
    case = ['1L']
    #case = ["SVM","SVMF","SVMW","NB","NBF","NBW","DT","Ada.Boost"]
    #case = ["MLP","RNN","LSTM"]

    cross_cv = 2
    tab_cross_cv = 1
    wave_type = wave_type_list[0]

    for filename in filename_list:
        case_list = []
        train_acc_list = []
        val_acc_list = []
        train_loss_list = []
        val_loss_list = []

        for each_case in case:
            if 1>0: #
                train_acc,val_acc,train_loss,val_loss = train(each_case,filename, cross_cv,tab_cross_cv,wave_type)
                case_list.append(case_label[each_case])
                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

                #visualize.epoch_acc_plotting(filename,case_list,FLAGS.sequence_window,tab_cross_cv,FLAGS.learning_rate,train_acc_list,val_acc_list)
                #visualize.epoch_loss_plotting(filename, case_list,FLAGS.sequence_window, tab_cross_cv, FLAGS.learning_rate,train_loss_list, val_loss_list)
            else:

                sys.stdout = tempstdout
                #sclearn.Basemodel(each_case,filename,cross_cv,tab_cross_cv)
                nnkeras.Basemodel(each_case,filename,cross_cv,tab_cross_cv)

    end = time.time()
    pprint("The time elapsed :  " + str(end - start) + ' seconds.\n')


if __name__ == "__main__":
    global tempstdout
    tempstdout = sys.stdout
    os.chdir("/home/cheng/Dropbox/code/MSLSTM")
    print("JHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHJ")
    print(os.getcwd())
    pprint("------------------------------------------------"+str(datetime.datetime.now())+"--------------------------------------------")
    start = time.time()
    tf.app.run()
