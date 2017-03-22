import time
import evaluation
from sklearn.feature_selection import RFE
from collections import defaultdict
import numpy as np
np.random.seed(1337)  # for reproducibility
import keras
from numpy import *
from sklearn import tree
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Dense
from keras.models import Model
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import tensorflow as tf
import loaddata
import printlog
import sys
import os
flags = tf.app.flags
import matplotlib.pyplot as plt
#import ucr_load_data

FLAGS = flags.FLAGS

def pprint(msg,method=''):
    if not 'Warning' in msg:
        #sys.stdout = printlog.PyLogger('',method)
        print(msg)
        try:
            sys.stderr.write(msg+'\n')
        except:
            pass
def Basemodel(_model,filename,cross_cv,tab_crosscv):

    result_list_dict = defaultdict(list)
    evaluation_list = ["AUC", "G_MEAN","ACCURACY","F1_SCORE"]
    for each in evaluation_list:
        result_list_dict[each] = []
    # num_selected_features = 25#AS leak tab=0
    # num_selected_features = 32#Slammer tab=0
    num_selected_features = 33  # Nimda tab=1
    x_train, y_train, x_val, y_val, x_test, y_test = loaddata.get_data_withoutS(FLAGS.pooling_type, FLAGS.is_add_noise, FLAGS.noise_ratio, FLAGS.data_dir,
                                            filename, FLAGS.sequence_window, tab_crosscv, cross_cv,
                                            multiScale=False, waveScale=FLAGS.scale_levels,
                                            waveType=FLAGS.wave_type)




    FLAGS.sequence_window = x_train.shape[1]
    FLAGS.input_dim = x_train.shape[-1]
    FLAGS.number_class = y_train.shape[1]
    FLAGS.batch_size = int(y_train.shape[0])

    for tab_cv in range(2):
        if tab_cv == tab_crosscv: continue
        # using MLP to train
        if _model == "MLP":
            #x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(FLAGS.is_add_noise, FLAGS.noise_ratio,
            #                                                                                FLAGS.data_dir, filename,
            #                                                                                FLAGS.sequence_window, tab_crosscv,
            #                                                                                cross_cv,
            #                                                                                Multi_Scale=FLAGS.is_multi_scale,
            #                                                                                Wave_Let_Scale=FLAGS.scale_levels,
            #                                                                                Normalize=0)

            pprint(_model + " is running..............................................")
            start = time.clock()
            model = Sequential()
            model.add(Dense(FLAGS.num_neurons1, activation="relu", input_dim=FLAGS.input_dim))

            model.add(Dense(output_dim=FLAGS.number_class))
            model.add(Activation("sigmoid"))
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
            # result = model.predict(X_Testing, batch_size=FLAGS.batch_size)
            result = model.predict(x_test)
            end = time.clock()
            pprint("The Time For MLP is " + str(end - start))

        elif _model == "RNN":
            pprint(_model + " is running..............................................")
            start = time.clock()
            x_train, y_train, x_val, y_val, x_test, y_test = loaddata.get_data(FLAGS.pooling_type, FLAGS.is_add_noise,
                                                                               FLAGS.noise_ratio, FLAGS.data_dir,
                                                                               filename, FLAGS.sequence_window,
                                                                               tab_crosscv, cross_cv,
                                                                               multiScale=False,
                                                                               waveScale=FLAGS.scale_levels,
                                                                               waveType=FLAGS.wave_type)

            rnn_object1 = SimpleRNN(FLAGS.num_neurons1, input_length=len(x_train[0]), input_dim=FLAGS.input_dim)
            model = Sequential()
            model.add(rnn_object1)  # X.shape is (samples, timesteps, dimension)
            #model.add(Dense(30, activation="sigmoid"))
            #rnn_object2 = SimpleRNN(FLAGS.num_neurons2, input_length=len(x_train[0]), input_dim=FLAGS.input_dim)
            #model.add(rnn_object2)  # X.shape is (samples, timesteps, dimension)
            model.add(Dense(output_dim=FLAGS.number_class))
            model.add(Activation("sigmoid"))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
            result = model.predict(x_test)
            end = time.clock()
            pprint("The Time For RNN is " + str(end - start))

            # print(result)
        elif _model == "LSTM":
            pprint(_model + " is running..............................................")
            start = time.clock()
            x_train, y_train, x_val, y_val, x_test, y_test = loaddata.get_data(FLAGS.pooling_type, FLAGS.is_add_noise,
                                                                               FLAGS.noise_ratio, FLAGS.data_dir,
                                                                               filename, FLAGS.sequence_window,
                                                                               tab_crosscv, cross_cv,
                                                                               multiScale=False,
                                                                               waveScale=FLAGS.scale_levels,
                                                                               waveType=FLAGS.wave_type)

            initi_weight = keras.initializers.RandomNormal(mean=0.0, stddev= 1, seed=None)
            initi_bias = keras.initializers.Constant(value=0.1)
            lstm_object = LSTM(FLAGS.num_neurons1, input_length=x_train.shape[1], input_dim=FLAGS.input_dim)
            model = Sequential()
            model.add(lstm_object,kernel_initializer=initi_weight,bias_initializer=initi_bias)  # X.shape is (samples, timesteps, dimension)
            #model.add(Dense(30, activation="relu"))
            model.add(Dense(output_dim=FLAGS.number_class))
            model.add(Activation("sigmoid"))
            sgd = keras.optimizers.SGD(lr=0.02, momentum=0.0, decay=0.0, nesterov=False)
            model.compile(optimizer=sgd,loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train,validation_data=(x_val, y_val), batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
            result = model.predict(x_test)
            end = time.clock()
            pprint("The Time For LSTM is " + str(end - start))

        results = evaluation.evaluation(y_test, result)  # Computing ACCURACY,F1-score,..,etc
        y_test2 = np.array(evaluation.ReverseEncoder(y_test))
        result2 = np.array(evaluation.ReverseEncoder(result))
        # Statistics False Alarm Rate
        if tab_cv == 2:
            with open(os.path.join(FLAGS.output,"StatFalseAlarm_" + filename + "_True"), "w") as fout:
                for tab in range(len(y_test2)):
                    fout.write(str(int(y_test2[tab])) + '\n')
            with open(os.path.join(FLAGS.output,"StatFalseAlarm_" + filename + "_" + _model + "_" + "_Predict"), "w") as fout:
                for tab in range(len(result2)):
                    fout.write(str(int(result2[tab])) + '\n')

        #for each_eval, each_result in results.items():
            #result_list_dict[each_eval].append(each_result)
        for each_eval in evaluation_list:
            result_list_dict[each_eval].append(results[each_eval])
    #for eachk, eachv in result_list_dict.items():
        #result_list_dict[eachk] = np.average(eachv)
    if FLAGS.is_add_noise == False:
        with open(os.path.join(FLAGS.output, "Comparison_Log_" + filename), "a")as fout:
            outfileline = _model + ":__"
            fout.write(outfileline)
            for each_eval in evaluation_list:
                fout.write(each_eval + ": " + str(round(np.average(result_list_dict[each_eval]), 3)) + ",\t")
            #for eachk, eachv in result_list_dict.items():
                #fout.write(eachk + ": " + str(round(eachv, 3)) + ",\t")
            fout.write('\n')

    return results
    # return epoch_training_loss_list,epoch_val_loss_list
