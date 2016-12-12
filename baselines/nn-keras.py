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

flags = tf.app.flags
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS


def Basemodel(_model,filename,cross_cv):
    filepath = FLAGS.data_dir
    sequence_window = FLAGS.sequence_window
    number_class = FLAGS.number_class
    hidden_units = FLAGS.num_neurons1
    input_dim = FLAGS.input_dim
    learning_rate = FLAGS.learning_rate
    epoch = FLAGS.max_epochs
    is_multi_scale = FLAGS.is_multi_scale
    training_level = FLAGS.scale_levels
    is_add_noise = FLAGS.is_add_noise
    noise_ratio = FLAGS.noise_ratio

    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY", "F1_SCORE", "AUC", "G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []
    np.random.seed(1337)  # for reproducibility
    # num_selected_features = 25#AS leak tab=0
    # num_selected_features = 32#Slammer tab=0
    num_selected_features = 33  # Nimda tab=1
    for tab_cv in range(cross_cv):
        # using MLP to train
        if _model == "MLP":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise, noise_ratio,
                                                                                            filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(_model + " is running..............................................")
            batch_size = len(y_train)
            start = time.clock()
            model = Sequential()
            model.add(Dense(hidden_units, activation="relu", input_dim=33))

            model.add(Dense(output_dim=number_class))
            model.add(Activation("sigmoid"))
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)
            # result = model.predict(X_Testing, batch_size=batch_size)
            result = model.predict(x_test)
            end = time.clock()
            print("The Time For MLP is " + str(end - start))

        elif _model == "RNN":
            print(_model + " is running..............................................")
            start = time.clock()
            x_train_multi_list, x_train, y_train, x_testing_multi_list, x_test, y_test = loaddata.GetData(is_add_noise,
                                                                                                          noise_ratio,
                                                                                                          'Attention',
                                                                                                          filepath,
                                                                                                          filename,
                                                                                                          sequence_window,
                                                                                                          tab_cv,
                                                                                                          cross_cv,
                                                                                                          Multi_Scale=is_multi_scale,
                                                                                                          Wave_Let_Scale=training_level)

            batch_size = len(y_train)
            rnn_object = SimpleRNN(hidden_units, input_length=len(x_train[0]), input_dim=input_dim)
            model = Sequential()
            model.add(rnn_object)  # X.shape is (samples, timesteps, dimension)
            model.add(Dense(30, activation="sigmoid"))
            model.add(Dense(output_dim=number_class))
            model.add(Activation("sigmoid"))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)
            result = model.predict(x_test)
            end = time.clock()
            print("The Time For RNN is " + str(end - start))

            # print(result)
        elif _model == "LSTM":
            print(_model + " is running..............................................")
            start = time.clock()
            x_train_multi_list, x_train, y_train, x_testing_multi_list, x_test, y_test = loaddata.GetData(is_add_noise,
                                                                                                          noise_ratio,
                                                                                                          'Attention',
                                                                                                          filepath,
                                                                                                          filename,
                                                                                                          sequence_window,
                                                                                                          tab_cv,
                                                                                                          cross_cv,
                                                                                                          Multi_Scale=is_multi_scale,
                                                                                                          Wave_Let_Scale=training_level)

            batch_size = len(y_train)
            lstm_object = LSTM(hidden_units, input_length=len(x_train[0]), input_dim=input_dim)
            model = Sequential()
            model.add(lstm_object)  # X.shape is (samples, timesteps, dimension)
            model.add(Dense(30, activation="relu"))
            model.add(Dense(output_dim=number_class))
            model.add(Activation("sigmoid"))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)
            result = model.predict(x_test)
            end = time.clock()
            print("The Time For LSTM is " + str(end - start))

        results = evaluation.evaluation(y_test, result)  # Computing ACCURACY,F1-score,..,etc
        y_test2 = np.array(evaluation.ReverseEncoder(y_test))
        result2 = np.array(evaluation.ReverseEncoder(result))
        # Statistics False Alarm Rate
        if tab_cv == 2:
            with open("StatFalseAlarm_" + filename + "_True.txt", "w") as fout:
                for tab in range(len(y_test2)):
                    fout.write(str(int(y_test2[tab])) + '\n')
            with open("StatFalseAlarm_" + filename + "_" + _model + "_" + "_Predict.txt", "w") as fout:
                for tab in range(len(result2)):
                    fout.write(str(int(result2[tab])) + '\n')

        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)

    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    if is_add_noise == False:
        with open(os.path.join(os.getcwd(), "Comparison_Log_" + filename + ".txt"), "a")as fout:
            outfileline = _model + ":__"
            fout.write(outfileline)
            for eachk, eachv in result_list_dict.items():
                fout.write(eachk + ": " + str(round(eachv, 3)) + ",\t")
            fout.write('\n')

    return results
    # return epoch_training_loss_list,epoch_val_loss_list
