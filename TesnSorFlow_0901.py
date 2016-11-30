import os
import time
import numpy as np
import math

#import tflearn
import LoadData
import Evaluation
import tensorflow as tf
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from collections import defaultdict
#Parameters
global filepath,sequence_window,number_class,batch_size,hidden_units,input_dim,epoch
filepath = os.getcwd()
sequence_window = 32
batch_size=1000
number_class = 2
hidden_units = 200
input_dim=33
epoch = 20
fixed_seed_num=1337
def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])
def Model(Label):
    global filepath,sequence_window, number_class, batch_size, hidden_units, input_dim
    cross_cv = 2
    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY","F1_SCORE","AUC","G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []

    for tab_cv in range(cross_cv):
        #if not tab_cv == 0 :continue
        x_train, y_train, x_test, y_test = LoadData.GetData(filepath, 'HB_Code_Red_I.txt', sequence_window,tab_cv,cross_cv)

        #using Keras to train
        if Label == "Keras":
            np.random.seed(fixed_seed_num)  # for reproducibility
            start = time.clock()
            lstm_object = LSTM(hidden_units, input_length=len(x_train[0]), input_dim=input_dim)

            model = Sequential()

            model.add(lstm_object)  # X.shape is (samples, timesteps, dimension)
            # model.add(LSTM(lstm_size,return_sequences=True,input_shape=(len(X_Training[0]),33)))
            # model.add(LSTM(100,return_sequences=True))
            #model.add(Dense(10, activation="tanh"))
            # model.add(Dense(5,activation="tanh"))

            model.add(Dense(output_dim=number_class))
            #model.add(Activation("sigmoid"))
            model.add(Activation("softmax"))

            # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)

            # result = model.predict(X_Testing, batch_size=batch_size)

            result = model.predict(x_test)

            end = time.clock()
            print("The Time For LSTM is " + str(end - start))
            #print(result)


        elif Label == "TensorFlow":
            tf.set_random_seed(fixed_seed_num)
            tf.reset_default_graph()
            # Network building
            num_neurons = hidden_units
            num_layers = 3
            dropout = tf.placeholder(tf.float32)
            data = tf.placeholder(tf.float32, [None,len(x_train[0]), input_dim])
            target = tf.placeholder(tf.float32, [None, number_class])

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
            #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)

            #lstm_cell = tf.nn.rnn_cell.GRUCell(num_neurons)
            val, state = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)
            weight = tf.Variable(tf.truncated_normal([num_neurons, int(target.get_shape()[1])]))
            bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

            prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
            #prediction = tf.nn.sigmoid(tf.matmul(last, weight) + bias)

            cross_entropy = -tf.reduce_sum(target * tf.log(prediction))


            optimizer = tf.train.AdamOptimizer()
            minimize = optimizer.minimize(cross_entropy)
            mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
            error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


            init_op = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init_op)

            no_of_batches = int(len(x_train) / batch_size)
            for i in range(epoch):
                ptr = 0
                for j in range(no_of_batches):
                    inp, out = x_train[ptr:ptr + batch_size], y_train[ptr:ptr + batch_size]
                    ptr += batch_size
                    sess.run(minimize, {data: inp, target: out})
                print "Epoch -------------------- ", str(i)
            #incorrect = sess.run(error, {data: x_test, target: y_test})
            result = sess.run(prediction, {data: x_test, target: y_test})
            #print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
            sess.close()

        results = Evaluation.Evaluation(y_test, result)

        print(results)

        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)


    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    print(result_list_dict)
#Model("Keras")
Model("TensorFlow")

"""
TensorFlow epoch=20
defaultdict(<type 'list'>, {'F1_SCORE': 0.83288000000000006, 'AUC': 0.82707192127476659, 'G_MEAN': 0.81058186700020962, 'ACCURACY': 0.82748500000000003})
defaultdict(<type 'list'>, {'F1_SCORE': 0.85648000000000002, 'AUC': 0.84449111143059175, 'G_MEAN': 0.82999842580447636, 'ACCURACY': 0.84546499999999991})


TensorFlow epoch=500
defaultdict(<type 'list'>, {'F1_SCORE': 0.86746499999999993, 'AUC': 0.85883619741162043, 'G_MEAN': 0.84658193736942056, 'ACCURACY': 0.85958999999999997})


"""

"""
Keras epoch=20
defaultdict(<type 'list'>, {'F1_SCORE': 0.87203999999999993, 'AUC': 0.8400284891925911, 'G_MEAN': 0.81316422400336541, 'ACCURACY': 0.84289499999999995})
Keras epoch=500
defaultdict(<type 'list'>, {'F1_SCORE': 0.89375500000000008, 'AUC': 0.87047613039893668, 'G_MEAN': 0.85389833380979629, 'ACCURACY': 0.87285999999999997})
"""

