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

def Model(Label,Parameters=[]):
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv
    try:
        filepath = Parameters["filepath"]
        filename = Parameters["filename"]
        sequence_window = Parameters["sequence_window"]
        number_class = Parameters["number_class"]
        hidden_units = Parameters["hidden_units"]
        input_dim = Parameters["input_dim"]
        learning_rate = Parameters["learning_rate"]
        epoch = Parameters["epoch"]
        is_multi_scale = Parameters["is_multi_scale"]
        training_level = Parameters["training_level"]
        cross_cv = Parameters["cross_cv"]
        fixed_seed_num = Parameters["fixed_seed_num"]
    except:
        pass



    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY","F1_SCORE","AUC","G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []

    for tab_cv in range(cross_cv):
        if not tab_cv == 0 :continue

        #print(is_multi_scale)
        x_train_multi_list,x_train, y_train, x_testing_multi_list,x_test, y_test = LoadData.GetData('Boosting',filepath, filename, sequence_window,tab_cv,cross_cv,Multi_Scale=is_multi_scale,Wave_Let_Scale=training_level)
        batch_size = len(y_train)
        if training_level < 0:
            pass
        else:
            x_train = x_train_multi_list
            x_test = x_testing_multi_list
        #using Keras to train
        if Label == "Keras-LSTM":

            np.random.seed(fixed_seed_num)  # for reproducibility
            start = time.clock()
            lstm_object = LSTM(hidden_units, input_length=sequence_window, input_dim=input_dim)
            model = Sequential()
            model.add(lstm_object)  # X.shape is (samples, timesteps, dimension)
            # model.add(LSTM(lstm_size,return_sequences=True,input_shape=(len(X_Training[0]),33)))
            # model.add(LSTM(100,return_sequences=True))
            # model.add(Dense(10, activation="tanh"))
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
        elif Label == "Keras-RNN":

            np.random.seed(fixed_seed_num)  # for reproducibility
            start = time.clock()
            rnn_object = SimpleRNN(hidden_units, input_length=sequence_window, input_dim=input_dim)
            model = Sequential()
            model.add(rnn_object)  # X.shape is (samples, timesteps, dimension)

            # model.add(Dense(10, activation="tanh"))
            # model.add(Dense(5,activation="tanh"))
            model.add(Dense(output_dim=number_class))

            #model.add(Activation("sigmoid"))
            model.add(Activation("softmax"))

            # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("________")
            print(x_train)
            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)

            # result = model.predict(X_Testing, batch_size=batch_size)
            result = model.predict(x_test)
            end = time.clock()
            print("The Time For RNN is " + str(end - start))
            # print(result)

        elif Label == "MS-LSTMB":
            print(Label+" is running..............................................")
            tf.reset_default_graph()
            tf.set_random_seed(fixed_seed_num)
            # Network building
            num_neurons = hidden_units

            if training_level > 0:
                number_of_scales = len(x_train_multi_list)
                scale_weight_init_value = 1.0/number_of_scales

                #scale_weight = tf.Variable(tf.constant(scale_weight_init_value,shape=[number_of_scales]), name="scale_weight")

                scale_weight = tf.Variable(tf.constant(scale_weight_init_value,shape=[1]), name="scale_weight")

                training_error = tf.Variable(tf.constant(0.,shape=[number_of_scales]), name="training_error")
                #training_error = tf.Variable(tf.constant(0.,shape=[]), name="training_error")


                total_error = tf.Variable(tf.constant(0.,shape=[]), name="total_error")

                #current_scale = tf.placeholder(tf.int32,shape=None)
                current_scale = tf.Variable(tf.constant(0,shape=[]))

                #indices = tf.constant([current_scale])

                #temp_ = tf.constant(1.0, shape=[number_of_scales])
                #scale_weight = tf.div(tf.exp(tf.mul(temp_,scale_weight)),tf.gather(tf.cumsum(tf.exp(tf.mul(temp_,scale_weight))),number_of_scales-1))
                #scale_weight = tf.div(tf.mul(temp_, scale_weight),tf.gather(tf.cumsum(tf.mul(temp_, scale_weight)), number_of_scales - 1))

                #output_scale_weight = tf.Print(scale_weight, [tf.exp(tf.sub(total_error,training_error))], message="This is Scale_Weight: ",first_n=1024, summarize=10)
                output_scale_weight = tf.Print(scale_weight, [scale_weight], message="This is Scale_Weight: ",first_n=1024, summarize=10)
                output_training_error = tf.Print(training_error, [training_error], message="This is Training Error: ",first_n=1024, summarize=10)
                output_total_error = tf.Print(total_error, [total_error], message="This is Total Error: ",first_n=1024, summarize=10)
                output_error_bwt_training_total = tf.Print(total_error, [tf.sub(total_error,training_error)], message="This is Error Bwt Training and Total: ",first_n=1024, summarize=10)

                data_original_train = tf.placeholder(tf.float32,[number_of_scales, batch_size, sequence_window, input_dim])
                data_for_lstm = tf.reshape(tf.gather(data_original_train,current_scale),[batch_size,sequence_window,input_dim])
                #data_original_train2 = tf.reshape(data_original_train,(number_of_scales, int(data_original_train.get_shape()[1]) * sequence_window * input_dim))
                print("At first this operation is assigned initial weights to each scale level, and this operation only runs once, after it will update...")

                scale_weight = tf.mul(scale_weight,tf.exp(tf.sub(total_error,tf.gather(training_error,current_scale))))
                #scale_weight = tf.mul(scale_weight,tf.exp(tf.sub(total_error,training_error)))
                print(tf.gather(training_error,0).get_shape())

                data_for_lstm = tf.mul(scale_weight,data_for_lstm)
                #data_for_lstm = tf.reshape(batch_vm2(scale_weight, data_original_train2),(int(data_original_train.get_shape()[1]), sequence_window, input_dim))
                #out_put_before_multi_first_level = tf.Print(data_original_train, [tf.gather(tf.gather(tf.gather(data_original_train, 0), 150), 0)],\
                                       # message="this is output for first level before multiply weight: ",\
                                       # first_n=1024, summarize=10)
                #data_for_lstm_multi_scale = data_for_lstm
                #output_data_for_lstm_multi_scale = tf.Print(data_for_lstm_multi_scale, [tf.gather(tf.gather(data_for_lstm_multi_scale, 0), 0)], message="This is data_for_lstm_multi_scale: ",first_n=1024, summarize=10)


                """
                #data_for_lstm_multi_scale = data_original_train
                out_put_before_multi_first_level = tf.Print(data_original_train, [tf.gather(tf.gather(tf.gather(data_original_train, 0), 150), 0)],\
                                        message="this is output for first level before multiply weight: ",\
                                        first_n=1024, summarize=10)
                out_put_before_multi_second_level = tf.Print(data_original_train, [tf.gather(tf.gather(tf.gather(data_original_train, 1), 150), 0)],\
                                        message="this is output for second level before multiply weight: ",\
                                        first_n=1024, summarize=10)
                # out_put_before_multi_third_level= tf.Print(data_for_lstm_0, [tf.gather(tf.gather(data_original_train, 2),0)],
                # message="this is output for third level before multiply weight: ")
                out_put_after_multi_level = tf.Print(data_for_lstm_0, [tf.gather(tf.gather(data_for_lstm_0, 150), 0)],\
                                                message="this is output for multi level after multiply weight: ",\
                                                first_n=1024, summarize=10)
                """

            else:
                data_for_lstm = tf.placeholder(tf.float32, [batch_size,sequence_window,input_dim])

            target = tf.placeholder(tf.float32, [None, number_class])
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
            #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
            #lstm_cell = tf.nn.rnn_cell.GRUCell(num_neurons)

            print(data_for_lstm.get_shape())

            val, state = tf.nn.dynamic_rnn(lstm_cell, data_for_lstm, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0]) - 1)

            weight = tf.Variable(tf.truncated_normal([num_neurons, int(target.get_shape()[1])]))
            bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
            #if training_level > 0:
                #prediction = tf.div(tf.nn.softmax(tf.matmul(last, weight) + bias),2.72)
            #else:
            prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
            #prediction = tf.nn.sigmoid(tf.matmul(last, weight) + bias)
            #output_weight = tf.Print(weight, [weight], message="This is Weight: ")
            #target = tf.exp(target)

            #cost_cross_entropy = -tf.reduce_mean(target * tf.log(prediction))
            #cost_cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target, name=None))  # Sigmoid
            cost_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target, name=None))  # Sigmoid

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            minimize = optimizer.minimize(cost_cross_entropy)

            mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
            error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
            output_correct_pred = tf.Print(prediction, [cost_cross_entropy],message="this is output for output_correct_pred : ",first_n=4096, summarize=40)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            init_op = tf.initialize_all_variables()
            sess = tf.Session()

            sess.run(init_op)


            no_of_batches = int(len(y_train) / batch_size)
            epoch_training_acc_list = []
            epoch_val_acc_list = []
            epoch_training_loss_list = []
            epoch_val_loss_list = []

            for i in range(epoch):
                ptr = 0
                for j in range(no_of_batches):
                    inp, out = x_train[ptr:ptr + batch_size], y_train[ptr:ptr + batch_size]
                    inp2, out2 = x_test[ptr:ptr + batch_size], y_test[ptr:ptr + batch_size]
                    inp = np.array(inp)
                    #print(inp.shape)
                    #sess.run(out_put_val, {data_original_train: inp, target: out})
                    #sess.run(out_put_Weight_W, {data_original_train: inp, target: out})
                    #sess.run(out_put_u_current_levels_temp, {data_original_train: inp, target: out})
                    #sess.run(out_put_u_current_levels_total, {data_original_train: inp, target: out})
                    #sess.run(out_put_u_current_levels, {data_original_train: inp, target: out})
                    #sess.run(out_put_m_total, {data_original_train: inp, target: out})
                    #sess.run(out_put_prediction, {data_original_train: inp, target: out})
                    ptr += batch_size
                    scale_acc_list = []
                    scale_error_list = []
                    for each_scale in range(number_of_scales):
                        sess.run(tf.assign(current_scale,int(each_scale)))
                        sess.run(data_for_lstm,{data_original_train: inp,target: out})
                        sess.run(minimize, {data_original_train: inp,target: out})
                        training_acc,training_loss = sess.run((accuracy,cost_cross_entropy),{data_original_train: inp,target: out})

                        #sess.run(tf.assign(tf.slice(training_error,0,each_scale),training_loss))
                        #sess.run(scale_weight,)


                        #sess.run(tf.scatter_update(training_error, tf.constant([each_scale]), tf.constant([float(training_loss)])),{data_original_train: inp,target: out})

                        #print("11111111")
                        #sess.run(output_scale_weight, {data_original_train: inp, target: out,current_scale:each_scale})
                        sess.run(output_correct_pred,{data_original_train: inp,target: out})


                        #sess.run(out_put_before_multi_first_level,{data_original_train: inp, target: out})
                        #sess.run(output_data_for_lstm_multi_scale,{data_original_train: inp, target: out})
                        scale_acc_list.append(training_acc)
                        scale_error_list.append(training_loss)

                        #sess.run(out_put_before_multi_first_level,{data_original_train: inp, target: out})
                        #sess.run(out_put_before_multi_second_level,{data_original_train: inp, target: out})
                        #sess.run(out_put_before_multi_third_level,{data_original_train: inp, target: out})
                        #sess.run(out_put_after_multi_level,{data_original_train: inp, target: out})

                        #sess.run(tf.assign(training_error, 1.0))
                    epoch_training_acc_list.append(np.mean(scale_acc_list))
                    epoch_training_loss_list.append(np.mean(scale_error_list))
                    sess.run(tf.assign(total_error,epoch_training_loss_list[-1]))
                    #sess.run(output_total_error, {data_original_train: inp, target: out, current_scale: each_scale})

                    #print("------------------"+str(i+1)+" epoches")
                    #print(scale_error_list)
                    #print(epoch_training_loss_list)
                    #for each_scale in range(number_of_scales):
                        #print(scale_error_list[each_scale])
                        #sess.run(tf.assign(training_error,scale_error_list[each_scale]))
                        #sess.run(tf.assign(training_error,1.0))

                        #sess.run(tf.assign(total_error,epoch_training_loss_list[-1]))

#--------------------------------------------------------------------------------------------

                    scale_acc_list = []
                    scale_error_list = []
                    for each_scale in range(number_of_scales):
                        sess.run(tf.assign(current_scale,int(each_scale)))
                        sess.run(data_for_lstm, {data_original_train: inp2, target: out2})
                        val_acc, val_loss = sess.run((accuracy, cost_cross_entropy),{data_original_train: inp2, target: out2})

                        scale_acc_list.append(val_acc)
                        scale_error_list.append(val_loss)


                    epoch_val_acc_list.append(np.mean(scale_acc_list))
                    epoch_val_loss_list.append(np.mean(scale_error_list))

                print("Epoch %s"%(str(i+1))+">"*20+"="+"train_accuracy: %s, train_loss: %s"%(str(epoch_training_acc_list[-1]),str(epoch_training_loss_list[-1]))\
                      +",\tval_accuracy: %s, val_loss: %s"%(str(epoch_val_acc_list[-1]),str(epoch_val_loss_list[-1])))

            #incorrect = sess.run(error, {data: x_test, target: y_test})
            #print("x_rest shape is ...")
            #print(x_test.shape)
            result_temp_scale = []
            for each_scale in range(number_of_scales):
                sess.run(tf.assign(current_scale, int(each_scale)))
                sess.run(data_for_lstm, {data_original_train: inp2, target: out2})
                result_temp = sess.run(prediction,{data_original_train: inp2, target: out2})

                #print(" result temp is ")
                #print(result_temp)
                result_temp_scale.append(result_temp)
            result_temp_scale = np.array(result_temp_scale)

            #print(" result temp scale is ")
            #print(result_temp_scale.shape)


            result = np.divide(np.sum(result_temp_scale,axis = 0),number_of_scales)

            #result = sess.run(prediction, {data_original_train: x_test, target: y_test})
            #print(result)
            #print("shape is ("+str(len(result))+","+str(len(result[0]))+')')
            #print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
            #if training_level > 0:
                #scale_weight = sess.run(scale_weight, {data_original_train: x_test, target: y_test})
                #print("The final scale weight is :\n")
                #print(scale_weight)
            #save_path = saver.save(sess, os.path.join(os.getcwd(),"modelckpt.txt"))
            #aaa = saver.restore(sess, os.path.join(os.getcwd(),"modelckpt.txt"))
            #all_variables = tf.trainable_variables()
            #var = [v for v in tf.trainable_variables() if v.name == "scale_weight"]
            sess.close()

        elif Label == "MS-LSTMA":
            pass

        #print(" y test ")
        #print(y_test)
        #print(" result is ")
        #print(result)
        #print(result)
        if len(Parameters) > 0:
            return Evaluation.Evaluation(y_test, result)#Plotting AUC

        results = Evaluation.Evaluation(y_test, result)#Computing ACCURACY, F1-Score, .., etc
        print(results)
        y_test2 = np.array(Evaluation.ReverseEncoder(y_test))
        result2 = np.array(Evaluation.ReverseEncoder(result))
        print("---------------------------1111111111111111")
        with open("StatFalseAlarm_"+filename+"_True.txt","w") as fout:
            for tab in range(len(y_test2)):
                fout.write(str(int(y_test2[tab]))+'\n')
        with open("StatFalseAlarm_"+filename+"_"+Label+"_"+"_Predict.txt","w") as fout:
            for tab in range(len(result2)):
                fout.write(str(int(result2[tab]))+'\n')
        print(result2.shape)
        print("---------------------------22222222222222222")
        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)


    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    print(result_list_dict)
    with open(os.path.join(os.getcwd(),"TensorFlow_Log"+filename+".txt"),"a")as fout:
        if is_multi_scale:
            outfileline = Label+"_____epoch:"+str(epoch)+",_____learning rate:"+str(learning_rate)+",_____multi_scale:"+str(is_multi_scale)+"\n"
        else:
            outfileline = Label+"_____epoch:"+str(epoch)+",_____learning rate:"+str(learning_rate)+",_____multi_scale:"+str(is_multi_scale)+",_____train_set_using_level:"+str(training_level)+"\n"

        fout.write(outfileline)
        for eachk,eachv in result_list_dict.items():
            fout.write(eachk+": "+str(round(eachv,3))+",\t")
        fout.write('\n')

    return epoch_training_loss_list,epoch_val_loss_list

def epoch_loss_plotting(train_loss_list,val_loss_list):
    num = len(train_loss_list)
    epoch = len(train_loss_list[0])
    x = [i+1 for i in range(epoch)]
    plt.plot(x,train_loss_list[1],'r-',label='multi-scale training loss')
    plt.plot(x,val_loss_list[1],'b-',label='multi-scale val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(x, val_loss_list[0], 'r-', label='original val loss')
    plt.plot(x, val_loss_list[1], 'g-', label='multi-scale val loss')
    plt.plot(x, val_loss_list[2], 'b-', label='multi-original val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()

#Model("Keras")
if __name__=='__main__':
    # Parameters
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv
    # ---------------------------Fixed Parameters------------------------------
    filepath = os.getcwd()
    #print(filepath)

    sequence_window = 10
    hidden_units = 200
    input_dim = 33
    number_class = 2
    cross_cv = 2
    fixed_seed_num = 1337
    # -------------------------------------------------------------------------
    filename = 'HB_Code_Red_I.txt'
    learning_rate = 0.01
    epoch = 50
    case_list = [1]
    multi_scale_value = sequence_window
    #case_list = [0]

    train_loss_list = [[] for i in range(3)]
    val_loss_list = [[] for i in range(3)]
    for each_case in case_list:
        if each_case == 0:
            is_multi_scale = False
            training_level = -1
        elif each_case == 1:
            is_multi_scale = True
            training_level = multi_scale_value
        elif each_case == 2:
            is_multi_scale = False
            training_level = multi_scale_value

        a,b = Model("MS-LSTMB")

        #a,b = Model("Keras-LSTM")
        #a,b = Model("Keras-RNN")

        train_loss_list[each_case].extend(a)
        val_loss_list[each_case].extend(b)

    #epoch_loss_plotting(train_loss_list,val_loss_list)