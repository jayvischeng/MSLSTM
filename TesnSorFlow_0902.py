import os
import time
import numpy as np
#import tflearn
import LoadData
import Evaluation
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from collections import defaultdict
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

def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

def Model(Label):
    global filepath, filename, sequence_window, number_class, batch_size, hidden_units, input_dim, learning_rate, epoch, multi_scale, training_level, cross_cv
    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY","F1_SCORE","AUC","G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []

    for tab_cv in range(cross_cv):
        #if not tab_cv == 0 :continue

        x_train_multi_list,x_train, y_train, x_testing_multi_list,x_test, y_test = LoadData.GetData(filepath, filename, sequence_window,tab_cv,cross_cv,Multi_Scale=True)

        if multi_scale:
            x_train = x_train_multi_list
            x_test = x_testing_multi_list
        else:
            if training_level < 0:
                pass
            else:
                x_train = x_train_multi_list[training_level]
            pass
        number_of_scales = len(x_train_multi_list)
        #using Keras to train
        if Label == "Keras":
            np.random.seed(fixed_seed_num)  # for reproducibility
            start = time.clock()
            lstm_object = LSTM(hidden_units, input_length=sequence_window, input_dim=input_dim)
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
            tf.reset_default_graph()
            tf.set_random_seed(fixed_seed_num)
            # Network building

            num_neurons = hidden_units

            num_layers = 3
            dropout = tf.placeholder(tf.float32)
            scale_weight = tf.Variable(tf.random_normal(shape=[number_of_scales]),name="scale_weight")
            #scale_weight = tf.Variable(np.array([0.25,0.25,0.25,0.25],dtype='f'))

            a = tf.Print(scale_weight, [scale_weight], message="This is Scale_Weight: ")


            data_original_train = tf.placeholder(tf.float32, [number_of_scales,batch_size,sequence_window,input_dim])
            #data_for_lstm_0 = tf.placeholder(tf.float32,[batch_size,sequence_window,input_dim],name="data_for_lstm_0")
            #data_for_lstm = tf.Variable(tf.zeros((batch_size,sequence_window,input_dim)),name="data_for_lstm")
            #data_for_lstm2 = tf.Variable(tf.zeros((batch_size,sequence_window,input_dim)),name="data_for_lstm2")
            #data_for_lstm = data_for_lstm_0
            #data_for_lstm = np.dot(data_original_train)

            data_original_train2 = tf.reshape(data_original_train,(number_of_scales,batch_size*sequence_window*input_dim))
            #for scale in range(number_of_scales):
                #data_for_lstm += tf.mul(scale_weight[scale], tf.gather(data_original_train, scale))
                #data_for_lstm_0 = tf.assign(data_for_lstm_0,data_for_lstm_0)
            data_for_lstm = tf.reshape(batch_vm2(scale_weight,data_original_train2),(batch_size,sequence_window,input_dim))
            if multi_scale:
                data_for_lstm_multi_scale = data_for_lstm
                #data_for_lstm_multi_scale = data_original_train
            else:
                data_original_train = tf.placeholder(tf.float32, [batch_size,sequence_window,input_dim])
                data_for_lstm_multi_scale = data_original_train


            target = tf.placeholder(tf.float32, [None, number_class])
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
            #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
            #lstm_cell = tf.nn.rnn_cell.GRUCell(num_neurons)

            val, state = tf.nn.dynamic_rnn(lstm_cell, data_for_lstm_multi_scale, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)

            weight = tf.Variable(tf.truncated_normal([num_neurons, int(target.get_shape()[1])]))
            bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
            #print(weight)
            prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
            #prediction = tf.nn.sigmoid(tf.matmul(last, weight) + bias)
            b = tf.Print(weight, [weight], message="This is Weight: ")

            cost_cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
            #cost_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target, name=None))  # Sigmoid

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            minimize = optimizer.minimize(cost_cross_entropy)

            #mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
            #error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            #copy_first_variable = data_for_lstm


            init_op = tf.initialize_all_variables()
            sess = tf.Session()

            #init_new_vars_op = tf.initialize_variables(var_list=[data_for_lstm_1])
            #sess.run(init_new_vars_op)

            #saver = tf.train.Saver()
            saver = tf.train.Saver({"my_interest": scale_weight})
            sess.run(init_op)

            no_of_batches = int(len(y_train) / batch_size)
            #print(no_of_batches)
            epoch_training_loss_list = []
            epoch_val_loss_list = []


            for i in range(epoch):
                #zeros_val = np.zeros(([batch_size,sequence_window,input_dim]),dtype='f')
                #sess.run(copy_first_variable,{data_for_lstm_0:zeros_val})
                #print(type(data_for_lstm))
                #data_for_lstm.initializer.run()
                ptr = 0
                for j in range(no_of_batches):
                    sess.run(a)
                    inp, out = x_train[ptr:ptr + batch_size], y_train[ptr:ptr + batch_size]
                    ptr += batch_size
                    sess.run(minimize, {data_original_train: inp, target: out})
                    training_acc,training_loss = sess.run((accuracy,cost_cross_entropy),{data_original_train: inp, target: out})
                    #training_loss = sess.run(cost_cross_entropy,{data_original_train: inp, target: out})
                    epoch_training_loss_list.append(training_loss)
                    val_acc,val_loss = sess.run((accuracy,cost_cross_entropy),{data_original_train: x_test, target: y_test})
                    #val_loss = sess.run(cost_cross_entropy, {data_original_train: x_test, target: y_test})
                    epoch_val_loss_list.append(val_loss)
                print("Epoch %s"%(str(i))+">"*20+"="+"train_accuracy: %s, train_loss: %s"%(str(training_acc),str(training_loss))\
                      +",\tval_accuracy: %s, val_loss: %s"%(str(val_acc),str(val_loss)))

            #incorrect = sess.run(error, {data: x_test, target: y_test})
            result = sess.run(prediction, {data_original_train: x_test, target: y_test})
            print(result)
            print("shape is ("+str(len(result))+","+str(len(result[0]))+')')
            #print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

            if multi_scale:
                scale_weight = sess.run(scale_weight, {data_original_train: x_test, target: y_test})
                print("The final scale weight is :\n")
                print(scale_weight)
            #save_path = saver.save(sess, os.path.join(os.getcwd(),"modelckpt.txt"))
            #aaa = saver.restore(sess, os.path.join(os.getcwd(),"modelckpt.txt"))
            #all_variables = tf.trainable_variables()
            #var = [v for v in tf.trainable_variables() if v.name == "scale_weight"]
            sess.close()

        results = Evaluation.Evaluation(y_test, result)
        print(results)
        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)


    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    print(result_list_dict)
    with open(os.path.join(os.getcwd(),"TensorFlow_Log"+filename+".txt"),"a")as fout:
        if multi_scale:
            outfileline = Label+"_____epoch:"+str(epoch)+",_____learning rate:"+str(learning_rate)+",_____multi_scale:"+str(multi_scale)+"\n"
        else:
            outfileline = Label+"_____epoch:"+str(epoch)+",_____learning rate:"+str(learning_rate)+",_____multi_scale:"+str(multi_scale)+",_____train_set_using_level:"+str(training_level)+"\n"

        fout.write(outfileline)
        for eachk,eachv in result_list_dict.items():
            fout.write(eachk+": "+str(round(eachv,3))+",\t")
        fout.write('\n')
    return epoch_training_loss_list,epoch_val_loss_list
def epoch_loss_plotting(train_loss_list,val_loss_list):
    num = len(train_loss_list)
    epoch = len(train_loss_list[0])
    x = [i+1 for i in range(epoch)]
    plt.plot(x,train_loss_list[0],'r-',label='multi-scale training loss')
    plt.plot(x,val_loss_list[0],'b-',label='multi-scale val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(x, val_loss_list[0], 'r-', label='multi-scale val loss')
    plt.plot(x, val_loss_list[1], 'g-', label='original val loss')
    plt.plot(x, val_loss_list[2], 'b-', label='scale 1 val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()

#Model("Keras")
if __name__=='__main__':
    # Parameters
    global filepath, filename, sequence_window, number_class, batch_size, hidden_units, input_dim, learning_rate, epoch, multi_scale, training_level, cross_cv
    # ---------------------------Fixed Parameters------------------------------
    filepath = os.getcwd()
    sequence_window = 32
    batch_size = 1888
    hidden_units = 200
    learning_rate = 0.005
    input_dim = 33
    number_class = 2
    cross_cv = 2
    fixed_seed_num = 1337
    # -------------------------------------------------------------------------
    filename = 'HB_Slammer.txt'
    epoch = 20
    case_list = [0]
    train_loss_list = [[] for i in range(len(case_list))]
    val_loss_list = [[] for i in range(len(case_list))]
    for each_case in case_list:
        if each_case == 0:
            multi_scale = True
        elif each_case == 1:
            multi_scale = False
            training_level = -1
        elif each_case == 2:
            multi_scale = False
            training_level = 0

        a,b = Model("TensorFlow")
        train_loss_list[each_case].extend(a)
        val_loss_list[each_case].extend(b)
    epoch_loss_plotting(train_loss_list,val_loss_list)


"""
TensorFlow epoch=20
defaultdict(<type 'list'>, {'F1_SCORE': 0.83288000000000006, 'AUC': 0.82707192127476659, 'G_MEAN': 0.81058186700020962, 'ACCURACY': 0.82748500000000003})
defaultdict(<type 'list'>, {'F1_SCORE': 0.85648000000000002, 'AUC': 0.84449111143059175, 'G_MEAN': 0.82999842580447636, 'ACCURACY': 0.84546499999999991})
defaultdict(<type 'list'>, {'F1_SCORE': 0.768625, 'AUC': 0.75246896555771037, 'G_MEAN': 0.7064861706432144, 'ACCURACY': 0.75385499999999994})
defaultdict(<type 'list'>, {'F1_SCORE': 0.81284500000000004, 'AUC': 0.79576681698554408, 'G_MEAN': 0.76700140720685783, 'ACCURACY': 0.79709000000000008})
defaultdict(<type 'list'>, {'F1_SCORE': 0.81641999999999992, 'AUC': 0.80378545378794763, 'G_MEAN': 0.77882093955102882, 'ACCURACY': 0.80479499999999993})

defaultdict(<type 'list'>, {'F1_SCORE': 0.81805000000000005, 'AUC': 0.77412265903042865, 'G_MEAN': 0.74937883387222992, 'ACCURACY': 0.77910999999999997})
defaultdict(<type 'list'>, {'F1_SCORE': 0.81805000000000005, 'AUC': 0.77412265903042865, 'G_MEAN': 0.74937883387222992, 'ACCURACY': 0.77910999999999997})
defaultdict(<type 'list'>, {'F1_SCORE': 0.81805000000000005, 'AUC': 0.77412265903042865, 'G_MEAN': 0.74937883387222992, 'ACCURACY': 0.77910999999999997})
defaultdict(<type 'list'>, {'F1_SCORE': 0.83675999999999995, 'AUC': 0.80624121632128531, 'G_MEAN': 0.79333618588826071, 'ACCURACY': 0.80993000000000004})


defaultdict(<type 'list'>, {'F1_SCORE': 0.89176999999999995, 'AUC': 0.90233722871452415, 'G_MEAN': 0.89703648611918141, 'ACCURACY': 0.89983000000000002})
defaultdict(<type 'list'>, {'F1_SCORE': 0.89176999999999995, 'AUC': 0.90233722871452415, 'G_MEAN': 0.89703648611918141, 'ACCURACY': 0.89983000000000002})
defaultdict(<type 'list'>, {'F1_SCORE': 0.83350000000000002, 'AUC': 0.85726210350584309, 'G_MEAN': 0.84529533715245708, 'ACCURACY': 0.85360000000000003})

TensorFlow epoch=500
defaultdict(<type 'list'>, {'F1_SCORE': 0.86746499999999993, 'AUC': 0.85883619741162043, 'G_MEAN': 0.84658193736942056, 'ACCURACY': 0.85958999999999997})


"""

"""
Keras epoch=20

defaultdict(<type 'list'>, {'F1_SCORE': 0.87203999999999993, 'AUC': 0.8400284891925911, 'G_MEAN': 0.81316422400336541, 'ACCURACY': 0.84289499999999995})
defaultdict(<type 'list'>, {'F1_SCORE': 0.79496, 'AUC': 0.72847100175746926, 'G_MEAN': 0.67597485420312675, 'ACCURACY': 0.73545000000000005})
defaultdict(<type 'list'>, {'F1_SCORE': 0.79496, 'AUC': 0.72847100175746926, 'G_MEAN': 0.67597485420312675, 'ACCURACY': 0.73545000000000005})
defaultdict(<type 'list'>, {'F1_SCORE': 0.85509000000000002, 'AUC': 0.8226276365706171, 'G_MEAN': 0.80436737525504476, 'ACCURACY': 0.82704999999999995})


defaultdict(<type 'list'>, {'F1_SCORE': 0.94911999999999996, 'AUC': 0.95158597662771283, 'G_MEAN': 0.95035359380360407, 'ACCURACY': 0.95033999999999996})
defaultdict(<type 'list'>, {'F1_SCORE': 0.94911999999999996, 'AUC': 0.95158597662771283, 'G_MEAN': 0.95035359380360407, 'ACCURACY': 0.95033999999999996})

Keras epoch=500
defaultdict(<type 'list'>, {'F1_SCORE': 0.89375500000000008, 'AUC': 0.87047613039893668, 'G_MEAN': 0.85389833380979629, 'ACCURACY': 0.87285999999999997})
"""



"""
defaultdict(<type 'list'>, {'F1_SCORE': 0.79888000000000003, 'AUC': 0.81767503542811548, 'G_MEAN': 0.81135318647966836, 'ACCURACY': 0.81506999999999996})



"""

