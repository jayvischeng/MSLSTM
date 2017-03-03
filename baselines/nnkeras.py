import time
import evaluation
from sklearn.feature_selection import RFE
from collections import defaultdict
import numpy as np
from numpy import *
from sklearn import tree
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Dense
from keras.models import Model
from keras.layers.recurrent import LSTM, SimpleRNN
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
    evaluation_list = ["ACCURACY", "F1_SCORE", "AUC", "G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []
    np.random.seed(1337)  # for reproducibility
    # num_selected_features = 25#AS leak tab=0
    # num_selected_features = 32#Slammer tab=0
    num_selected_features = 33  # Nimda tab=1
    for tab_cv in range(cross_cv):
        if tab_cv == tab_crosscv: continue
        np.random.seed(1337)  # for reproducibility
        # using MLP to train
        if _model == "MLP":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(FLAGS.is_add_noise, FLAGS.noise_ratio,
                                                                                            FLAGS.data_dir, filename,
                                                                                            FLAGS.sequence_window, tab_crosscv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=FLAGS.is_multi_scale,
                                                                                            Wave_Let_Scale=FLAGS.scale_levels,
                                                                                            Normalize=0)
            #x_train, y_train, x_test, y_test = ucr_load_data.load_ucr_data(FLAGS.is_multi_scale,filename)
            FLAGS.sequence_window = x_train.shape[len(x_train.shape) - 2]
            # FLAGS.sequence_window = 900
            FLAGS.input_dim = x_train.shape[-1]
            FLAGS.number_class = y_train.shape[1]
            FLAGS.batch_size = int(y_train.shape[0] / 2)
            if FLAGS.is_multi_scale:
                FLAGS.scale_levels = x_train.shape[1]
                FLAGS.sequence_window = x_train.shape[len(x_train.shape) - 2]
                FLAGS.input_dim = x_train.shape[-1]
                FLAGS.number_class = y_train.shape[1]
                FLAGS.batch_size = int(y_train.shape[0] / 2)
            else:
                FLAGS.sequence_window = x_train.shape[1]
                FLAGS.input_dim = x_train.shape[-1]
                FLAGS.number_class = y_train.shape[1]
                FLAGS.batch_size = int(y_train.shape[0])
            #FLAGS.sequence_window = 270
            pprint(_model + " is running..............................................")
            start = time.clock()
            model = Sequential()
            model.add(Dense(FLAGS.num_neurons1, activation="relu", input_dim=FLAGS.input_dim))

            model.add(Dense(output_dim=FLAGS.number_class))
            model.add(Activation("sigmoid"))
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
            # result = model.predict(X_Testing, batch_size=FLAGS.batch_size)
            result = model.predict(x_test)
            end = time.clock()
            pprint("The Time For MLP is " + str(end - start))

        elif _model == "RNN":
            pprint(_model + " is running..............................................")
            start = time.clock()
            x_train, y_train, x_test, y_test = loaddata.GetData(FLAGS.pooling_type,FLAGS.is_add_noise,
                                                                FLAGS.noise_ratio,
                                                                'Attention',
                                                                FLAGS.data_dir,
                                                                filename,
                                                                FLAGS.sequence_window,
                                                                tab_crosscv,
                                                                cross_cv,
                                                                Multi_Scale=False,
                                                                Wave_Let_Scale=FLAGS.scale_levels)

            rnn_object = SimpleRNN(FLAGS.num_neurons1, input_length=len(x_train[0]), input_dim=FLAGS.input_dim)
            model = Sequential()
            model.add(rnn_object)  # X.shape is (samples, timesteps, dimension)
            model.add(Dense(30, activation="sigmoid"))
            model.add(Dense(output_dim=FLAGS.number_class))
            model.add(Activation("sigmoid"))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
            result = model.predict(x_test)
            end = time.clock()
            pprint("The Time For RNN is " + str(end - start))

            # print(result)
        elif _model == "LSTM":
            pprint(_model + " is running..............................................")
            start = time.clock()
            x_train, y_train, x_test, y_test = loaddata.GetData(FLAGS.pooling_type,FLAGS.is_add_noise,
                                                                FLAGS.noise_ratio,
                                                                'Attention',
                                                               FLAGS.data_dir,
                                                                filename,
                                                                FLAGS.sequence_window,
                                                                tab_crosscv,
                                                                cross_cv,
                                                                Multi_Scale=False,
                                                                Wave_Let_Scale=FLAGS.scale_levels)

            #x_train, y_train, x_test, y_test = ucr_load_data.load_ucr_data(FLAGS.is_multi_scale,filename)
            FLAGS.sequence_window = x_train.shape[len(x_train.shape) - 2]
            FLAGS.sequence_window = 90
            FLAGS.input_dim = x_train.shape[-1]
            FLAGS.number_class = y_train.shape[1]
            FLAGS.batch_size = int(y_train.shape[0] / 2)
            if FLAGS.is_multi_scale:
                FLAGS.scale_levels = x_train.shape[1]
                FLAGS.sequence_window = x_train.shape[len(x_train.shape) - 2]
                FLAGS.input_dim = x_train.shape[-1]
                FLAGS.number_class = y_train.shape[1]
                FLAGS.batch_size = int(y_train.shape[0] / 2)
            else:
                FLAGS.sequence_window = x_train.shape[1]
                FLAGS.input_dim = x_train.shape[-1]
                FLAGS.number_class = y_train.shape[1]
                FLAGS.batch_size = int(y_train.shape[0])
            lstm_object = LSTM(FLAGS.num_neurons1, input_length=len(x_train[0]), input_dim=FLAGS.input_dim)
            model = Sequential()
            model.add(lstm_object)  # X.shape is (samples, timesteps, dimension)
            model.add(Activation('tanh'))
            #model.add(Dense(30, activation="relu"))
            model.add(Dense(output_dim=FLAGS.number_class))
            model.add(Activation("softmax"))
            #model.compile(optimizer='adam', learning_rate=FLAGS.learning_rate,loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train,validation_data=(x_test, y_test), batch_size=FLAGS.batch_size, nb_epoch=FLAGS.max_epochs)
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

        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)

    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    if FLAGS.is_add_noise == False:
        with open(os.path.join(FLAGS.output, "Comparison_Log_" + filename), "a")as fout:
            outfileline = _model + ":__"
            fout.write(outfileline)
            for eachk, eachv in result_list_dict.items():
                fout.write(eachk + ": " + str(round(eachv, 3)) + ",\t")
            fout.write('\n')

    return results
    # return epoch_training_loss_list,epoch_val_loss_list
