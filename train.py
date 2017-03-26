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
import sklearn
from sklearn.metrics import confusion_matrix
from baselines import sclearn
import evaluation
from collections import defaultdict
import tensorflow as tf
import mslstm
import config
import loaddata
import numpy as np
import visualize
from sklearn.metrics import accuracy_score
from baselines import nnkeras,sclearn
import matplotlib.pyplot as plt
flags = tf.app.flags

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
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
def pprint(msg,method=''):
    #if not 'Warning' in msg:
    if 1>0:
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

def train_lstm(method,filename_train_list,filename_test,trigger_flag,evalua_flag,is_binary_class,result_list_dict,evaluation_list):
    global tempstdout
    FLAGS.option = method
    dropout = 0.8
    x_train, y_train, x_val, y_val, x_test, y_test = loaddata.get_data(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, FLAGS.data_dir,
                                           filename_test, FLAGS.sequence_window, trigger_flag,
                                            multiScale=FLAGS.is_multi_scale, waveScale=FLAGS.scale_levels,
                                            waveType=FLAGS.wave_type)
    """
    if filename_test == 'HB_AS_Leak.txt':
        filename_train = 'HB_C_N_S.txt'
    elif filename_test == 'HB_Code_Red_I.txt':
        filename_train = 'HB_A_N_S.txt'
    elif filename_test == 'HB_Nimda.txt':
        filename_train = 'HB_A_C_S.txt'
    elif filename_test == 'HB_Slammer.txt':
        filename_train = 'HB_A_C_N.txt'
    print(filename_test)
    #x_train, y_train, x_val, y_val = loaddata.get_trainData(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, FLAGS.data_dir,
    #                                                        filename_train, FLAGS.sequence_window, trigger_flag,
    #                                                        multiScale=FLAGS.is_multi_scale, waveScale=FLAGS.scale_levels,
    #                                                        waveType=FLAGS.wave_type)
    #x_test, y_test = loaddata.get_testData(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, FLAGS.data_dir,
    #                                       filename_test, FLAGS.sequence_window, trigger_flag,
    #                                        multiScale=FLAGS.is_multi_scale, waveScale=FLAGS.scale_levels,
    #                                        waveType=FLAGS.wave_type)

    """
    #loaddata.Multi_Scale_Plotting_2(x_train)

    print(x_test.shape)

    if FLAGS.is_multi_scale:
        FLAGS.scale_levels = x_train.shape[1]
        FLAGS.sequence_window = x_train.shape[len(x_train.shape) - 2]
        FLAGS.input_dim = x_train.shape[-1]
        FLAGS.number_class = y_train.shape[1]
        if "Nimda" in filename_test:
            FLAGS.batch_size = int(int(x_train.shape[0])/5)
        else:
            FLAGS.batch_size = int(x_train.shape[0])
    else:
        FLAGS.sequence_window = x_train.shape[1]
        FLAGS.input_dim = x_train.shape[-1]
        FLAGS.number_class = y_train.shape[1]
        if "Nimda" in filename_test:
            FLAGS.batch_size = int(int(x_train.shape[0])/5)
    #g = tf.Graph()
    with tf.Graph().as_default():
    #with tf.variable_scope("middle")as scope:
        tf.set_random_seed(1337)
        #global_step = tf.Variable(0,name="global_step",trainable=False)
        data_x,data_y = mslstm.inputs(FLAGS.option)
        #output_u_w,prediction, label = mslstm.inference(data_x,data_y,FLAGS.option)

        is_training = tf.placeholder(tf.bool)
        prediction, label,output_last = mslstm.inference(data_x,data_y,FLAGS.option,is_training)
        loss = mslstm.loss_(prediction, label)
        tran_op,optimizer = mslstm.train(loss)
        minimize = optimizer.minimize(loss)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #summary_op = tf.merge_all_summaries()
        weights = tf.Variable(tf.constant(0.1, shape=[len(y_test)*FLAGS.sequence_window, 1, FLAGS.scale_levels]),
                              name="weights123")
        init_op = tf.global_variables_initializer()
        #init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)

        #summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
        #saver = tf.train.Saver()
        saver = tf.train.Saver({"my_weights": weights})

        epoch_training_loss_list = []
        epoch_training_acc_list = []
        epoch_val_loss_list = []
        epoch_val_acc_list = []
        early_stopping = 10
        no_of_batches = int(len(x_train) / FLAGS.batch_size)
        #visualize.curve_plotting_withWindow(x_train, y_train, 0, "Train_"+'_'+FLAGS.option)
        #visualize.curve_plotting_withWindow(x_test, y_test, 2, "Test_"+'_'+FLAGS.option)
        total_iteration = 0
        for i in range(FLAGS.max_epochs):
            if early_stopping > 0:
                pass
            else:
                break
            j_iteration = 0
            for j_batch in iterate_minibatches(x_train,y_train,FLAGS.batch_size,shuffle=False):
                j_iteration += 1
                total_iteration += 1
                inp, out = j_batch
                sess.run(minimize, {data_x: inp, data_y: out, is_training:True})
                training_acc, training_loss = sess.run((accuracy, loss), {data_x: inp, data_y: out,is_training:True})
                #sys.stdout = tempstdout
                val_acc, val_loss = sess.run((accuracy, loss), {data_x:x_val, data_y:y_val,is_training:True})
            pprint(
                FLAGS.option + "_Epoch%s" % (str(i + 1)) + ">" * 3 +'_Titer-'+str(total_iteration) +'_iter-'+str(j_iteration)+ str(FLAGS.wave_type) + '-' + str(FLAGS.scale_levels) + '-' + str(FLAGS.learning_rate)+'-'+str(FLAGS.num_neurons1)+'-'+str(FLAGS.num_neurons2)+ ">>>=" + "train_accuracy: %s, train_loss: %s" % (
                str(training_acc), str(training_loss)) \
                + ",\tval_accuracy: %s, val_loss: %s" % (str(val_acc), str(val_loss)), method)


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
            if val_loss > 10 or val_loss == np.nan:
                break
        if 1>0:
            #pprint("PPP")
            weights_results = sess.run(output_last, {data_x:x_test, data_y: y_test})
            #print(weights_results)
            #sys.stdout = tempstdout
            visualize.curve_plotting(weights_results,y_test,filename_test,FLAGS.option)
            #pprint("QQQ")
            with open(filename_test+"_EA.txt",'w')as fout:
                fout.write(weights_results)
            #sess.run(weights.assign(weights_results))
        else:
            pass

        #weights = output_u_w.eval(session=sess)
        #weights = saver.restore(sess, "./tf_tmp/model.ckpt")
        #pprint(weights)
        #weight_list = return_max_index(weights)
        result = sess.run(prediction, {data_x:x_test, data_y: y_test})
        #print(result)
        #pprint(result)
        #print("LLL")
    saver.save(sess, "./tf_tmp/model.ckpt")
    sess.close()
    #results = evaluation.evaluation(y_test, result)#Computing ACCURACY, F1-Score, .., etc
    if is_binary_class == True:
        results = evaluation.evaluation(y_test, result, trigger_flag, evalua_flag)  # Computing ACCURACY,F1-score,..,etc
    else:
        symbol_list = [0, 1, 2, 3, 4]
        confmat = confusion_matrix(y_test, result, labels=symbol_list)
        visualize.plotConfusionMatrix(confmat)
        #accuracy = sklearn.metrics.accuracy_score(y_test, result)
        symbol_list2 = [0]
        y_ = []
        for symbol in symbol_list2:
            for tab in range(len(y_test)):
                if y_test[tab] == symbol and y_test[tab] == result[tab]:
                    y_.append(symbol)
            # print(y_test[0:10])
            # rint(result[0:10])
            # print("Accuracy is :"+str(accuracy))
            accuracy = float(len(y_)) / (list(result).count(symbol))
            print("Accuracy of " + str(symbol) + " is :" + str(accuracy))
        print("Accuracy is :" + str(accuracy))
        f1_score = sklearn.metrics.f1_score(y_test, result)
        print("F-score is :" + str(f1_score))
        results = {'ACCURACY': accuracy, 'F1_SCORE': f1_score, 'AUC': 9999, 'G_MEAN': 9999}
    sys.stdout = tempstdout
    #print(weights_results.shape)
    #print("215")
    y_test2 = np.array(evaluation.ReverseEncoder(y_test))
    result2 = np.array(evaluation.ReverseEncoder(result))
    #results = accuracy_score(y_test2, result2)
    #print(y_test2)
    #print(result2)
    #print(results)
    with open(os.path.join(os.path.join(os.getcwd(),'stat'),"StatFalseAlarm_" + filename_test + "_True.txt"), "w") as fout:
        for tab in range(len(y_test2)):
            fout.write(str(int(y_test2[tab])) + '\n')
    with open(os.path.join(os.path.join(os.getcwd(),'stat'),"StatFalseAlarm_" + filename_test + "_" + method + "_" + "_Predict.txt"), "w") as fout:
        for tab in range(len(result2)):
            fout.write(str(int(result2[tab])) + '\n')
    #eval_list = ["AUC", "G_MEAN","ACCURACY","F1_SCORE"]
    for each_eval in evaluation_list:
        result_list_dict[each_eval].append(results[each_eval])

    if evalua_flag:
        with open(os.path.join(FLAGS.output, "TensorFlow_Log" + filename_test + ".txt"), "a")as fout:
            if not FLAGS.is_multi_scale:
                outfileline = FLAGS.option  + "_epoch:" + str(FLAGS.max_epochs) + ",_lr:" + str(FLAGS.learning_rate) + ",_multi_scale:" + str(FLAGS.is_multi_scale) + ",hidden_nodes: "+str(FLAGS.num_neurons1)+"/"+str(FLAGS.num_neurons2) + "\n"
            else:
                outfileline = FLAGS.option  + "_epoch:" + str(FLAGS.max_epochs) + ",_wavelet:"+str(FLAGS.wave_type) + ",_lr:" + str(FLAGS.learning_rate) + ",_multi_scale:" + str(FLAGS.is_multi_scale) + ",_train_set_using_level:" + str(FLAGS.scale_levels) + ",hidden_nodes: "+str(FLAGS.num_neurons1)+"/"+str(FLAGS.num_neurons2) + "\n"

            fout.write(outfileline)
            for each_eval in evaluation_list:
            #for eachk, eachv in result_list_dict.items():
                fout.write(each_eval + ": " + str(round(np.mean(result_list_dict[each_eval]), 3)) + ",\t")
            fout.write('\n')
        return epoch_training_acc_list,epoch_val_acc_list,epoch_training_loss_list,epoch_val_loss_list
    else:
        return results




def train_classic(method,filename_train,filename_test, trigger_flag,evalua_flag,is_binary_class,evaluation_list):
    return sclearn.Basemodel(method,filename_train,filename_test,trigger_flag,evalua_flag,evaluation_list)

def train(method,filename_train,filename_test,trigger_flag,evalua_flag,is_binary_class,evaluation_list,wave_type='db1'):
    global data_x, data_y
    result_list_dict = defaultdict(list)
    #evaluation_list = ["ACCURACY", "F1_SCORE", "AUC", "G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []
    if 'L' in method or 'RNN' in method:
        sys.stdout = tempstdout
        if method == '1L' or method == '2L' or method == '3L' \
                or method == '4L' or method == '5L' or method == 'RNN':
            #FLAGS.learning_rate = 0.01
            FLAGS.is_multi_scale = False
        elif 'AL' == method:
            #FLAGS.learning_rate = 0.01
            FLAGS.is_multi_scale = False
        else:
            #FLAGS.learning_rate = 0.05
            FLAGS.is_multi_scale = True
            FLAGS.wave_type = wave_type
        return train_lstm(method,filename_train,filename_test,trigger_flag,evalua_flag,is_binary_class,result_list_dict,evaluation_list)
    else:
        sys.stdout = tempstdout
        return train_classic(method,filename_train,filename_test,trigger_flag,evalua_flag,is_binary_class,result_list_dict,evaluation_list)

def main(unused_argv):
    global tempstdout

    #main function


    #wave_type_list =['db1','db2','haar','coif1','db1','db2','haar','coif1','db1','db2']
    wave_type_list = ['haar']
    multi_scale_value_list = [2,3,4,5,6,10]
    case_label = {'SVM':'SVM','NB':'NB','DT':'DT','Ada.Boost':'Ada.Boost','RF':'RF','1NN':'1NN','1NN-DTW':'DTW',
                  'SVMF':'SVMF','SVMW':'SVMW','MLP':'MLP','RNN':'RNN','1L':'LSTM','2L':'2-LSTM','3L':'3-LSTM',\
                  'AL':'ALSTM','HL':'MSLSTM','HAL':'MSLSTM'}

    trigger_flag = 1
    evalua_flag = True
    is_binary_class = False
    single_layer = False

    if is_binary_class:
        filename_list = ["HB_AS_Leak.txt","HB_Code_Red_I.txt","HB_Nimda.txt","HB_Slammer.txt"]
        #filename_list = ["HB_Slammer.txt"]  # HB_Code_Red_I.txt
                                                # HB_Nimda.txt
                                                # HB_Slammer.txt
    else:
        filename_list = ["HB_ALL.txt"]

    if trigger_flag == 1 :
        if single_layer:
            #case = ["MLP"]
            #case = ['1L','3L']
            case = ['MLP','RNN','1L','2L','3L','AL']
        else:
            case = ['HAL']
            #case = ['HL','HAL']

    else:
        #case = ["1NN-DTW"]
        #case = ["RF","SVM","SVMF","SVMW","NB","DT","Ada.Boost","1NN"]
        case = ["SVM","NB","1NN","Ada.Boost","RF"]

    if evalua_flag:
        evaluation_list = ["AUC", "G_MEAN", "ACCURACY", "F1_SCORE"]
    else:
        evaluation_list = ["FPR", "TPR","AUC","G_MEAN"]

    wave_type = wave_type_list[0]
    #hidden_unit1_list = [8, 16, 32, 64, 100, 128, 200, 256]
    hidden_unit1_list = [32]

    #hidden_unit2_list = [8, 16, 20, 32, 64]
    hidden_unit2_list = [8]
    comnination_list = []
    for each1 in hidden_unit1_list:
        for each2 in hidden_unit2_list:
            comnination_list.append([each1, each2])
    #if single_layer:
        #comnination_list = hidden_unit1_list
    #learning_rate_list = [0.001, 0.01, 0.05, 0.1]
    learning_rate_list = [0.01]

    for tab in range(len(filename_list)):
        case_list = []
        train_acc_list = []
        val_acc_list = []
        train_loss_list = []
        val_loss_list = []

        results = {}
        for each_case in case:
            case_list.append(case_label[each_case])
            if trigger_flag: #
                sys.stdout = tempstdout
                if each_case == 'MLP':
                    if evalua_flag:
                        nnkeras.Basemodel(each_case, filename_list[tab],trigger_flag,evalua_flag,is_binary_class,evaluation_list)
                    else:
                        results[case_label[each_case]] = nnkeras.Basemodel(each_case, filename_list[tab],trigger_flag,evalua_flag,is_binary_class,evaluation_list)

                else:
                    if evalua_flag:
                        for learning_rate in learning_rate_list:
                            FLAGS.learning_rate = learning_rate
                            for each_comb in comnination_list:
                                #if single_layer:
                                    #FLAGS.num_neurons1 = each_comb[0]
                                #else:
                                    #FLAGS.num_neurons1, FLAGS.num_neurons2 = each_comb
                                FLAGS.num_neurons1, FLAGS.num_neurons2 = each_comb

                                train_acc,val_acc,train_loss,val_loss = train(each_case,filename_list, filename_list[tab],trigger_flag,evalua_flag,is_binary_class,evaluation_list,wave_type)
                                train_acc_list.append(train_acc)
                                val_acc_list.append(val_acc)
                                train_loss_list.append(train_loss)
                                val_loss_list.append(val_loss)
                                #visualize.epoch_acc_plotting(filename_list[tab],case_list,FLAGS.sequence_window,FLAGS.learning_rate,train_acc_list,val_acc_list)
                                #visualize.epoch_loss_plotting(filename_list[tab], case_list,FLAGS.sequence_window, FLAGS.learning_rate,train_loss_list, val_loss_list)
                    else:
                        results[case_label[each_case]] = train(each_case,filename_list, filename_list[tab],trigger_flag,evalua_flag,is_binary_class,evaluation_list,wave_type)

            else:
                sys.stdout = tempstdout
                if evalua_flag:
                    sclearn.Basemodel(each_case, filename_list[tab], trigger_flag, evalua_flag,is_binary_class,evaluation_list)
                else:
                    results[case_label[each_case]] = sclearn.Basemodel(each_case, filename_list[tab],trigger_flag,evalua_flag,is_binary_class,evaluation_list)

        if not evalua_flag:
            visualize.plotAUC(results,case_list,filename_list[tab])
        else:
            if trigger_flag:
                try:
                    visualize.epoch_acc_plotting(filename_list[tab], case_list, FLAGS.sequence_window,FLAGS.learning_rate, train_acc_list, val_acc_list)
                    visualize.epoch_loss_plotting(filename_list[tab], case_list, FLAGS.sequence_window,FLAGS.learning_rate, train_loss_list, val_loss_list)
                except:
                    pass
    end = time.time()
    pprint("The time elapsed :  " + str(end - start) + ' seconds.\n')


if __name__ == "__main__":
    global tempstdout
    tempstdout = sys.stdout
    pprint("------------------------------------------------"+str(datetime.datetime.now())+"--------------------------------------------")
    start = time.time()
    tf.app.run()
