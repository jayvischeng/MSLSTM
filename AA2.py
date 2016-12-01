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
import matplotlib
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()






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
def normalized_scale_levels(scales_list):
    return tf.div(scales_list,tf.gather(tf.gather(tf.cumsum(scales_list, axis=1), 0), int(scales_list.get_shape()[1]-1)))
    #print("hahaha")
    #print(scales_list.get_shape())
    #a0=tf.cumsum(scales_list)
    #a0 = tf.cumsum(tf.gather(scales_list,scales_list.get_shape()[0]-1))
    #print(a0.get_shape)
    #a1 = tf.cast(tf.gather(tf.cumsum(scales_list),scales_list.get_shape()[1]-1),tf.float32)
    #print(a1.get_shape())
    #a2 = tf.constant(tf.gather(a1,0),shape=scales_list.get_shape())
    #output2 = tf.Print(a2,[a2],message="2222: ")
    #a3 = tf.div(scales_list,a2)
    #a3 =scales_list
    #return a3,a1,a0,scales_list
    #print(tf.constant(tf.truediv(scales_list,a2),shape=a3.get_shape()))
    #print(tf.Variable(tf.constant(tf.truediv(scales_list,a2))))
    #return tf.Variable(tf.constant(tf.truediv(scales_list,a2)))
    #print(tf.constant(tf.cast(a2,tf.float32),shape=[1,scales_list.get_shape()[1]]))
    #denomier = tf.Variable(tf.constant(tf.gather(tf.cumsum(scales_list),scales_list.get_shape()[0]-1),dtype=tf.float32,shape=[1,scales_list.get_shape()[1]]))
    #return tf.truediv(scales_list,denomier)
    #return map(lambda a:float(a)/np.sum(scales_list),scales_list)

def Model(each_case,Label,Parameters=[]):
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv, wave_type, is_add_noise, noise_ratio, pooling_type,corss_val_label

    try:
        filepath = Parameters["filepath"]
        filename = Parameters["filename"]
        sequence_window = Parameters["sequence_window"]
        number_class = Parameters["number_class"]
        hidden_units = Parameters["hidden_units"]
        input_dim = Parameters["input_dim"]
        learning_rate = Parameters["learning_rate"]
        epoch = Parameters["epoch"]
        training_level = Parameters["training_level"]
        cross_cv = Parameters["cross_cv"]
        fixed_seed_num = Parameters["fixed_seed_num"]
        wave_type = Parameters["wave_type"]
        is_add_noise = Parameters["is_add_noise"]
        is_multi_scale = Parameters["is_multi_scale"]
        noise_ratio = Parameters["noise_ratio"]
        pooling_type = Parameters["pooling_type"]
    except:
        pass


    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY","F1_SCORE","AUC","G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []



    for tab_cv in range(cross_cv):
        if not tab_cv == corss_val_label: continue
        print("******************************"+str(tab_cv))
        #if corss_val_label == False:
            #if 'Nimda' in filename:
                #if not tab_cv == 1: continue
            #else:
            #if not tab_cv == 1 :continue#AS Leak, Code Red I, Slammer
        #else:
            #pass

        x_train, y_train,x_test, y_test = LoadData.GetData(pooling_type,is_add_noise,noise_ratio,'Attention',filepath, filename, sequence_window,tab_cv,cross_cv,Multi_Scale=is_multi_scale,Wave_Let_Scale=training_level,Wave_Type=wave_type)

        batch_size = 1
        if Label == "MS-LSTM":
            tf.reset_default_graph()
            tf.set_random_seed(fixed_seed_num)
            num_neurons = hidden_units
            # Network building
            if is_multi_scale == True and each_case == 2:
                number_scale_levels = training_level
                data_original_train = tf.placeholder(tf.float32,[number_scale_levels,batch_size,sequence_window,input_dim])

                u_w = tf.Variable(tf.random_normal(shape=[1,number_scale_levels]), name="u_w")

                data_original_train1 = tf.transpose(data_original_train,[1,2,3,0])
                print("aaa")
                print(data_original_train1.get_shape())
                max_pooling_output = tf.nn.max_pool(data_original_train1,[1,sequence_window,1,1], \
                                                    [1, 1, 1, 1],padding='VALID')

                print(max_pooling_output.get_shape())
                print("bbb")

                max_pooling_output_reshape = tf.reshape(max_pooling_output,(batch_size,input_dim,number_scale_levels))
                max_pooling_output2 = tf.transpose(max_pooling_output_reshape,[0,2,1])

                #data_original_train_merged = batch_vm2(data_original_train2,tf.transpose(u_w_scales_normalized))
                #data_original_train_merged = tf.reshape(data_original_train_merged,(batch_size,sequence_window,input_dim))
                #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
                #val_list, state_list = [tf.nn.dynamic_rnn(lstm_cell, tf.gather(data_original_train2,i), dtype=tf.float32) for i in range(number_scale_levels)]
                #val = tf.transpose(val,[1,0,2])
                #val2_list = [tf.gather(tf.gather(val_list,i),val.get_shape()[0]-1) for i in range(number_scale_levels)]
                #val = tf.reshape(val,[batch_size*number_of_scales,num_neurons])

                Weight_W = tf.Variable(tf.truncated_normal([input_dim,number_scale_levels]))
                b_W = tf.Variable(tf.constant(0.1, shape=[1,number_scale_levels]))
                batch_W = tf.Variable(tf.constant(0.1, shape=[batch_size,number_scale_levels]))

                #u_current_levels_temp = tf.matmul(tf.gather(max_pooling_output2,max_pooling_output2.get_shape()[0]-1),Weight_W)+b_W
                u_current_levels_temp = batch_vm(max_pooling_output2,Weight_W)+b_W
                print("ddd")
                print(u_current_levels_temp.get_shape())
                temp1 = tf.reshape((batch_vm(u_current_levels_temp,tf.transpose(u_w))),(batch_size,number_scale_levels))
                temp2 = batch_vm(u_current_levels_temp,tf.transpose(u_w))
                print(temp2.get_shape())
                u_current_levels_total = tf.gather(tf.cumsum(tf.exp(tf.transpose(temp1,[1,0]))),number_scale_levels-1)
                print("eee")
                #u_current_levels_total_list = tf.gather(tf.cumsum(tf.exp(tf.mul(tf.transpose(u_current_levels_temp,[0,1,2]),tf.transpose(u_w)))),number_scale_levels-1)
                #u_current_levels_total_list = tf.cumsum(tf.exp(tf.mul(tf.transpose(u_current_levels_temp,[0,1,2]),tf.transpose(u_w))))

                #print(u_current_levels_total_list.get_shape())
                u_current_levels = tf.transpose(tf.div(tf.transpose(temp1,[1,0]),u_current_levels_total),[1,0])
                print(u_current_levels.get_shape())

                out_put_u_w_scale = tf.Print(u_current_levels,[u_current_levels],"The u_current_levels shape is ----------------:",first_n=4096,summarize=40)

                #target = tf.placeholder(tf.float32, [batch_size, number_class])
                #m_total = batch_vm(tf.transpose(u_current_levels),val)
                #u_w_scales_normalized = u_current_levels
                #tf.assign(u_w_scales_normalized,u_current_levels)
                #m_total = tf.mul(tf.transpose(u_current_levels),val)
                #weight = tf.Variable(tf.truncated_normal([num_neurons, int(target.get_shape()[1])]))
                #bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
                #prediction = tf.nn.softmax(tf.matmul(m_total, weight) + bias)

                #u_w_scales_normalized = tf.Variable(tf.constant(1.0/number_scale_levels,shape=[1,number_scale_levels]), name="u_w")
                #u_w_scales_normalized = normalized_scale_levels(u_w_scales_normalized)
                #u_w = tf.Variable(tf.random_normal(shape=[1,sequence_window]), name="u_w")



                #output_data_original_train = tf.Print(data_original_train,[data_original_train],"The Original Train  is :",first_n=4096,summarize=40)

                #data_original_train = tf.placeholder(tf.float32,[batch_size,sequence_window,input_dim])

                u_w_AAA = tf.Variable(tf.random_normal(shape=[1,sequence_window]), name="u_w_AAA")

                data_original_train2 = tf.transpose(data_original_train,[1,2,3,0])

                print("ccc")
                print(data_original_train2.get_shape())
                print(tf.transpose(u_current_levels).get_shape())

                #u_current_levels = tf.reshape(u_current_levels,(batch_size,1,1,number_scale_levels))
                #data_original_train_merged = batch_vm2(data_original_train2,u_current_levels)

                data_original_train_merged = batch_vm2(data_original_train2,tf.transpose(u_current_levels))
                data_original_train_merged = tf.transpose(data_original_train_merged,[2,3,0,1])
                data_original_train_merged = tf.reshape(data_original_train_merged,(batch_size*batch_size,sequence_window,input_dim))

                index_list = [(i+1)*(i+1)-1 for i in range(batch_size)]
                data_original_train_merged = tf.gather(data_original_train_merged,index_list)

                #data_original_train_merged = tf.map_fn(batch_vm2,
                data_original_train_merged = tf.reshape(data_original_train_merged,(batch_size,sequence_window,input_dim))
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
                val, state = tf.nn.dynamic_rnn(lstm_cell, data_original_train_merged, dtype=tf.float32)

                #val = tf.transpose(val,[1,0,2])
                val2 = tf.gather(val,val.get_shape()[0]-1)
                #val = tf.reshape(val,[batch_size*number_of_scales,num_neurons])
                out_put_val = tf.Print(val,[val.get_shape()],"The val shape is :",first_n=4096,summarize=40)
                out_put_val2 = tf.Print(val2,[val2.get_shape()],"The val2 shape is :",first_n=4096,summarize=40)

                Weight_W_AAA = tf.Variable(tf.truncated_normal([num_neurons,sequence_window]))
                out_put_Weight_W_AAA = tf.Print(Weight_W_AAA,[Weight_W_AAA],"The Weight_W_AAA is :",first_n=1024,summarize=10)

                b_W_AAA = tf.Variable(tf.constant(0.1, shape=[sequence_window,sequence_window]))
                out_put_b_W_AAA = tf.Print(b_W_AAA,[b_W_AAA.get_shape()],"The b_W_AAA shape is :",first_n=1024,summarize=10)

                #tf.reshape(tf.matmul(tf.reshape(Aijk,[i*j,k]),Bkl),[i,j,l])

                #u_current_levels_temp = tf.reshape(tf.mul(tf.reshape(val,[batch_size*num_neurons],Weight_W)+b_W
                #print("val shape is ")
                #print(val2.get_shape())
                #print(Weight_W.get_shape())
                #print(b_W.get_shape())
                u_current_levels_temp_AAA = tf.matmul(val2,Weight_W_AAA)+b_W_AAA

                out_put_u_current_levels_b_W_AAA = tf.Print(b_W_AAA,[b_W_AAA],"The b_W shape is :",first_n=4096,summarize=40)
                out_put_u_current_levels_temp_AAA = tf.Print(u_current_levels_temp_AAA,[u_current_levels_temp_AAA],"The u_current_levels_temp  is :",first_n=4096,summarize=40)
                out_put_u_current_u_w_AAA = tf.Print(u_w,[u_w],"The u_w shape is :",first_n=4096,summarize=40)

                u_current_levels_total_AAA = tf.gather(tf.cumsum(tf.exp(batch_vm(u_current_levels_temp_AAA,tf.transpose(u_w_AAA)))),sequence_window-1)
                #print(tf.transpose(u_w).get_shape())
                out_put_u_current_levels_total_AAA = tf.Print(u_current_levels_total_AAA,[u_current_levels_total_AAA],"The u_current_levels_total_AAA shape is :",first_n=4096,summarize=40)

                u_current_levels_AAA = tf.div(tf.exp(batch_vm(u_current_levels_temp_AAA,tf.transpose(u_w_AAA))),u_current_levels_total_AAA)
                out_put_u_current_levels_AAA = tf.Print(u_current_levels_AAA,[u_current_levels_AAA],"The u_current_levels_AAA shape is :",first_n=4096,summarize=40)

                target = tf.placeholder(tf.float32, [batch_size, number_class])
                #print("-----------------------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                #print(val.get_shape())


                m_total = batch_vm(tf.transpose(u_current_levels_AAA),val)

                #u_w_scales_normalized = u_current_levels
                #tf.assign(u_w_scales_normalized,u_current_levels)
                #m_total = tf.mul(tf.transpose(u_current_levels),val)

                #print(m_total.get_shape())

                out_put_m_total_shape = tf.Print(m_total,[m_total.get_shape()],"The m_total shape is :",first_n=4096,summarize=40)
                out_put_m_total = tf.Print(m_total,[m_total],"The m_total  is :",first_n=4096,summarize=40)

                weight = tf.Variable(tf.truncated_normal([num_neurons, int(target.get_shape()[1])]))
                bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
                prediction = tf.nn.softmax(tf.matmul(m_total, weight) + bias)

                out_put_prediction = tf.Print(prediction,[prediction.get_shape()],"The prediction shape is :",first_n=1024,summarize=10)
                #print(prediction.get_shape())

            else:
                try:
                    number_scale_levels = training_level
                    u_w_scales_normalized = tf.Variable(tf.constant(1.0/number_scale_levels,shape=[1,number_scale_levels]), name="u_w")
                    u_w_scales_normalized = normalized_scale_levels(u_w_scales_normalized)
                    u_w = tf.Variable(tf.random_normal(shape=[1,sequence_window]), name="u_w")


                    data_original_train = tf.placeholder(tf.float32,[number_scale_levels,batch_size,sequence_window,input_dim])

                    output_data_original_train = tf.Print(data_original_train,[data_original_train],"The Original Train  is :",first_n=4096,summarize=40)

                    #data_original_train = tf.placeholder(tf.float32,[batch_size,sequence_window,input_dim])
                    data_original_train2 = tf.transpose(data_original_train,[1,2,3,0])
                    data_original_train_merged = batch_vm2(data_original_train2,tf.transpose(u_w_scales_normalized))
                    data_original_train_merged = tf.reshape(data_original_train_merged,(batch_size,sequence_window,input_dim))
                    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)

                    val, state = tf.nn.dynamic_rnn(lstm_cell, data_original_train_merged, dtype=tf.float32)

                    target = tf.placeholder(tf.float32, [batch_size, number_class])

                except:
                    data_original_train = tf.placeholder(tf.float32, [None,sequence_window,input_dim])
                    target = tf.placeholder(tf.float32, [None, number_class])
                    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_neurons, forget_bias=1.0, activation=tf.nn.tanh)
                    val, state = tf.nn.dynamic_rnn(lstm_cell, data_original_train, dtype=tf.float32)


                val = tf.transpose(val, [1, 0, 2])
                last = tf.gather(val, int(val.get_shape()[0]) - 1)

                weight = tf.Variable(tf.truncated_normal([num_neurons, int(target.get_shape()[1])]))
                bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

                prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)


            #cost_cross_entropy = -tf.reduce_mean(target * tf.log(prediction))
            cost_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, target, name=None))  # Sigmoid

            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            minimize = optimizer.minimize(cost_cross_entropy)

            #mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
            #error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            init_op = tf.initialize_all_variables()
            sess = tf.Session()

            sess.run(init_op)

            #batch_size = 1

            no_of_batches = int(len(y_train) / batch_size)
            epoch_training_loss_list = []
            epoch_training_acc_list = []
            epoch_val_loss_list = []
            epoch_val_acc_list = []
            weight_list=[]
            early_stopping = 10
            epoch_stop = epoch

            for i in range(epoch):
                if early_stopping > 0:
                    pass
                else:
                    epoch_stop = i+1
                    break
                ptr = 0
                for j in range(no_of_batches):
                    inp, out = x_train[:,ptr:ptr + batch_size], y_train[ptr:ptr + batch_size]
                    inp2, out2 = x_test[:,ptr:ptr + batch_size], y_test[ptr:ptr + batch_size]
                    if len(inp[0])!= batch_size:continue
                    #print("INPUT IS ")
                    #print(inp.shape)
                    #print("OUTPUT IS ")
                    #print(inp2.shape)
                    #da.Plotting_Sequence(inp,out)
                    try:
                        pass

                        #sess.run(out_put_u_w_scale, {data_original_train: inp, target: out})
                        #sess.run(output__1,{data_original_train: inp, target: out})
                        #sess.run(output0,{data_original_train: inp, target: out})
                        #sess.run(output1,{data_original_train: inp, target: out})

                        #print("11111")
                        #print(out_put_u_w_scale)
                        #print("22222")
                        #print(normalized_scale_levels(out_put_u_w_scale))
                        #print(normalized_scale_levels(out_put_u_w_scale).shape)
                        #sess.run(tf.assign(u_w_scales,normalized_scale_levels(out_put_u_w_scale)))

                        #sess.run(out_put_original_train, {data_original_train: inp, target: out})
                        #sess.run(out_put_val, {data_original_train: inp, target: out})
                        #sess.run(out_put_val2, {data_original_train: inp, target: out})
                        #sess.run(out_put_Weight_W, {data_original_train: inp, target: out})
                        #sess.run(out_put_u_current_levels_temp, {data_original_train: inp, target: out})
                        #sess.run(out_put_u_current_u_w, {data_original_train: inp, target: out})
                        #sess.run(out_put_u_current_levels_b_W, {data_original_train: inp, target: out})

                        #sess.run(out_put_u_current_levels_total, {data_original_train: inp, target: out})
                        #weight_list.append(sess.run(out_put_u_w_scale, {data_original_train: inp, target: out}))
                        #sess.run(out_put_m_total, {data_original_train: inp, target: out})
                        #sess.run(out_put_m_total_shape, {data_original_train: inp, target: out})

                        #sess.run(out_put_prediction, {data_original_train: inp, target: out})
                    except:
                        pass
                    #print(out)
                    ptr += batch_size
                    #print(inp.shape)
                    sess.run(minimize, {data_original_train: inp,target: out})
                    training_acc,training_loss = sess.run((accuracy,cost_cross_entropy),{data_original_train: inp, target: out})
                        #sess.run(out_put_before_multi_first_level,{data_original_train: inp, target: out})
                        #sess.run(output_data_for_lstm_multi_scale,{data_original_train: inp, target: out})

                    epoch_training_loss_list.append(training_loss)
                    epoch_training_acc_list.append(training_acc)
                        #sess.run(out_put_before_multi_first_level,{data_original_train: inp, target: out})
                        #sess.run(out_put_before_multi_second_level,{data_original_train: inp, target: out})
                        #sess.run(out_put_before_multi_third_level,{data_original_train: inp, target: out})
                        #sess.run(out_put_after_multi_level,{data_original_train: inp, target: out})

                    #sess.run(minimize, {data_original_train: inp2,target: out2})

                    val_acc,val_loss = sess.run((accuracy,cost_cross_entropy),{data_original_train: inp2, target: out2})
                    epoch_val_loss_list.append(val_loss)
                    epoch_val_acc_list.append(val_acc)
                print("Epoch %s"%(str(i+1))+">"*20+"="+"train_accuracy: %s, train_loss: %s"%(str(training_acc),str(training_loss))\
                      +",\tval_accuracy: %s, val_loss: %s"%(str(val_acc),str(val_loss)))
                try:
                    max_val_acc = epoch_val_acc_list[-2]
                except:
                    max_val_acc = 0

                if epoch_val_acc_list[-1] < max_val_acc:
                    early_stopping -= 1
                elif epoch_val_acc_list[-1] >= max_val_acc:
                    early_stopping = 100
            #incorrect = sess.run(error, {data: x_test, target: y_test})
            #print("x_test shape is ..."+str(x_test.shape))
            #print(x_test)
            try:
                weight_list = []
                result = []
                ptr = 0
                for t in range(int(len(y_test)/batch_size)):

                    inp_test, out_test = x_test[:,ptr:ptr + batch_size], y_test[ptr:ptr + batch_size]
                    if len(inp_test[0])!= batch_size:continue

                    result.append(sess.run(prediction, {data_original_train:inp_test, target: out_test}))
                    weight_list.append(sess.run(out_put_u_w_scale, {data_original_train: inp_test, target: out_test}))

                    ptr += batch_size
                x_ = [i for i in range(number_scale_levels)]
                y_ = [i for i in range(int(len(y_test)/batch_size))]
                x_,y_ = np.meshgrid(x_,y_)
                plt.plot(np.array(x_),np.array(y_),np.array(result))
                plt.show()



            except:
                pass
                #x_test = x_test[0:batch_size]
                #y_test = y_test[0:batch_size]
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

        elif Label == "MS-LSTMB":
            pass

        results = Evaluation.Evaluation(y_test, result)#Computing ACCURACY, F1-Score, .., etc

        try:
            for each_eval, each_result in results.items():
                result_list_dict[each_eval].append(each_result)
            if len(Parameters) > 0:
                label = "PW"
            else:
                label = "DA"
        except:
            label = "AUC"

        #if len(Parameters) > 0:
            #try:
                #for each_eval, each_result in results.items():
                    #result_list_dict[each_eval].append(each_result)
                #label = "PW"
                #with open(os.path.join(os.getcwd(), "TensorFlow_Log" + filename + ".txt"), "a")as fout:
                  #  if training_level > 0:
                   #     outfileline = Label + "_____epoch:" + str(epoch) + ",_____learning rate:" + str(learning_rate) + ",_____multi_scale:" + str(is_multi_scale) + "\n"
                    #else:
                     #   outfileline = Label + "_____epoch:" + str(epoch) + ",_____learning rate:" + str(learning_rate) + ",_____multi_scale:" + str(is_multi_scale) + ",_____train_set_using_level:" + str(training_level) + "\n"

                    #fout.write(outfileline)
                    #for eachk, eachv in result_list_dict.items():
                     #   fout.write(eachk + ": " + str(round(eachv, 3)) + ",\t")
                    #fout.write('\n')

                #return results
            #except:
                #label = "AUC"
                #return Evaluation.Evaluation(y_test, result)#Plotting AUC
        #else:
            #for each_eval, each_result in results.items():
                #result_list_dict[each_eval].append(each_result)
            #label = "da"


        #if label == "AUC": return results
        if label == "DA":
            pass
            """
            y_test2 = np.array(Evaluation.ReverseEncoder(y_test))
            result2 = np.array(Evaluation.ReverseEncoder(result))
            with open("StatFalseAlarm_"+filename+"_True.txt","w") as fout:
                for tab in range(len(y_test2)):
                    fout.write(str(int(y_test2[tab]))+'\n')
            with open("StatFalseAlarm_"+filename+"_"+Label+"_"+"_Predict.txt","w") as fout:
                for tab in range(len(result2)):
                    fout.write(str(int(result2[tab]))+'\n')
            """
    try:
        for eachk, eachv in result_list_dict.items():
            result_list_dict[eachk] = np.average(eachv)
        print(result_list_dict)
        if is_add_noise == False:
            if corss_val_label == 0:
                outputfilename = "Tab_A_MS-LSTM_Log_"+filename+".txt"

            else:
                outputfilename = "Tab_B_MS-LSTM_Log_"+filename+".txt"
            with open(os.path.join(os.getcwd(),outputfilename),"a")as fout:
                if training_level>0:
                    outfileline = Label+"_epoch:"+str(epoch_stop)+",__wavelet type:"+str(wave_type)+",__pooling type:"+str(pooling_type)+",__learning rate:"+str(learning_rate)+",__multi_scale:"+str(is_multi_scale)+",__scale_levels:"+str(training_level)+",__sequence_window:"+str(sequence_window)+"\n"
                else:
                    outfileline = Label+"_epoch:"+str(epoch_stop)+",__wavelet type:"+str(wave_type)+",__learning rate:"+str(learning_rate)+",__multi_scale:"+str(is_multi_scale)+",__scale_levels:"+str(training_level)+",__sequence_window:"+str(sequence_window)+"\n"

                fout.write(outfileline)
                for eachk,eachv in result_list_dict.items():
                    fout.write(eachk+": "+str(round(eachv,3))+",\t")
                fout.write('\n')
        else:
            with open(os.path.join(os.getcwd(), "MS-LSTM_Log_Adding_Noise_" + filename + ".txt"), "a")as fout:
                if training_level > 0:
                    outfileline = Label + "_____epoch:" + str(epoch_stop) +",_____pooling type:"+str(pooling_type)+ ",_____learning rate:" + \
                        str(learning_rate) + ",_____multi_scale:" + str(is_multi_scale) + "\n"
                else:
                    outfileline = Label + "_____epoch:" + str(epoch_stop) + ",_____pooling type:"+str(pooling_type)+ ",_____learning rate:" + \
                        str(learning_rate) + ",_____multi_scale:" + str(is_multi_scale) + ",_____train_set_using_level:" + str(training_level) + "\n"

                fout.write(outfileline)
                for eachk, eachv in result_list_dict.items():
                    fout.write(eachk + ": " + str(round(eachv, 3)) + ",\t")
                fout.write('\n')
    except:
        pass
    #print("lallala")
    #print(epoch_training_loss_list)
    if not "DA"==label: return results
    return epoch_training_loss_list,epoch_val_loss_list,epoch_training_acc_list,epoch_val_acc_list,weight_list,results


def epoch_acc_plotting(train_acc_list,val_acc_list):
    global filename,sequence_window,corss_val_label,learning_rate

    epoch = len(train_acc_list[0])
    x = [i+1 for i in range(epoch)]
    plt.figure()
    plt.plot(x,train_acc_list[0],'g-',label='LSTM Train Accuray')
    plt.plot(x,train_acc_list[1],'b-',label='MS-LSTM Train Accuray')
    plt.plot(x,train_acc_list[2],'r-',label='MS-LSTM-AT Train Accuray')

    plt.plot(x, val_acc_list[0], 'y-', label='LSTM Val Accuracy')
    plt.plot(x, val_acc_list[1], 'c-', label='MS-LSTM Val Accuracy')
    plt.plot(x, val_acc_list[2], 'm-', label='MS-LSTM-AT Val Accuracy')

    #plt.plot(x, val_loss_list[2], 'b-', label='multi-original val loss')
    plt.xlabel('Epoch',fontsize=12)
    if 'AS' in filename:
        plt.ylim(0.0,1.0)
    else:
        plt.ylim(0.05,1.05)
    plt.ylabel('Accuracy',fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.legend(loc='lower right',fontsize=10)
    #plt.legend(fontsize=10)
    plt.title(filename.split('.')[0].replace('HB_','')+'/sw: '+str(sequence_window)+"/lr: "+str(learning_rate))

    if corss_val_label == 0:
        plt.savefig("Tab_A_Epoch_ACC_"+filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".pdf",dpi=600)
        plt.savefig("Tab_A_Epoch_ACC_"+filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png",dpi=600)
    else:
        plt.savefig("Tab_B_Epoch_ACC_" + filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".pdf", dpi=600)
        plt.savefig("Tab_B_Epoch_ACC_" + filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png", dpi=600)

def epoch_loss_plotting(train_loss_list,val_loss_list):
    global filename,sequence_window,corss_val_label,learning_rate

    epoch = len(train_loss_list[0])
    x = [i+1 for i in range(epoch)]
    plt.figure()

    plt.plot(x,train_loss_list[0],'g-',label='LSTM Train Loss')
    plt.plot(x,train_loss_list[1],'b-',label='MS-LSTM Train Loss')
    plt.plot(x,train_loss_list[2],'r-',label='MS-LSTM-AT Train Loss')

    plt.plot(x, val_loss_list[0], 'y-', label='LSTM Val Loss')
    plt.plot(x, val_loss_list[1], 'c-', label='MS-LSTM Val Loss')
    plt.plot(x, val_loss_list[2], 'm-', label='MS-LSTM-AT Val Loss')


    plt.xlabel('Epoch',fontsize=12)
    plt.ylim(0.5,0.95)
    plt.ylabel('Loss',fontsize=12)
    plt.grid()
    plt.tick_params(labelsize=12)

    plt.legend(loc='upper right',fontsize=10)

    plt.title(filename.split('.')[0].replace('HB_','')+'/sw: '+str(sequence_window)+"/lr: "+str(learning_rate))

    if corss_val_label == 0:
        plt.savefig("Tab_A_Epoch_Loss_"+filename+"_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".pdf",dpi=600)
        plt.savefig("Tab_A_Epoch_Loss_"+filename+"_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png",dpi=600)
    else:
        plt.savefig("Tab_B_Epoch_Loss_" + filename + "_SW_" + str(sequence_window)+"_LR_"+str(learning_rate) + ".pdf", dpi=600)
        plt.savefig("Tab_B_Epoch_Loss_" + filename + "_SW_" + str(sequence_window)+"_LR_"+str(learning_rate) + ".png", dpi=600)

    #plt.show()

def weight_plotting(filename,weight_list):
    """
    if "Code" in filename:
        optimal_scale_weight = 8
    elif 'Nimda' in filename:
        optimal_scale_weight = 9
    elif 'AS_Leak' in filename:
        optimal_scale_weight =
    """
    global corss_val_label,learning_rate
    weight_list_new = np.transpose(weight_list)
    #a,b,c = weight_list_new.shape
    subtitle = ['a', 'b', 'c', 'd', 'e', 'f','g','h','i','j']

    X = [i for i in range(len(weight_list_new[0][0]))]
    plt.figure(figsize=(24,12),dpi=600)
    count = 0
    for tab in range(10):
        index = tab
        plt.subplot(1,4,count+1)
        if tab == 9:
            index = -1
        elif tab == 8:
            index = -2
        plt.plot(X,weight_list_new[0][index])
        plt.xlabel('Epoch\n('+subtitle[count]+') Scale '+str(tab+1), fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.grid()
        count += 1

    """
    plt.subplot(2,5,2)
    plt.plot(X,weight_list_new[0][1])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,3)
    plt.plot(X,weight_list_new[0][2])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,4)
    plt.plot(X,weight_list_new[0][3])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()


    plt.subplot(2,5,5)
    plt.plot(X,weight_list_new[0][4])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,6)
    plt.plot(X,weight_list_new[0][5])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,7)
    plt.plot(X,weight_list_new[0][6])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,8)
    plt.plot(X,weight_list_new[0][7])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,9)
    plt.plot(X,weight_list_new[0][10])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,10)
    plt.plot(X,weight_list_new[0][11])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()
    """


    #num = len(train_loss_list)
    #epoch = len(train_loss_list[0])
    #x = [i+1 for i in range(epoch)]
    #plt.figure()
    #plt.plot(x,train_loss_list[0],'r-',label='LSTM training loss')
    #plt.plot(x,train_loss_list[1],'b-',label='MS-LSTM training loss')
    #plt.plot(x, val_loss_list[0], 'm-', label='LSTM val loss')
    #plt.plot(x, val_loss_list[1], 'c-', label='MS-LSTM val loss')
    #plt.plot(x, val_loss_list[2], 'b-', label='multi-original val loss')
    #plt.tick_params(labelsize=8)
    #plt.legend(fontsize=8)
    plt.tight_layout()
    if corss_val_label == 0:
        plt.savefig("Tab_A_Weight_list_" + filename + "_SW_" + str(sequence_window) +"_LR_"+str(learning_rate) + ".pdf", dpi=600)
        plt.savefig("Tab_A_Weight_list_" + filename + "_SW_" + str(sequence_window) +"_LR_"+str(learning_rate) + ".png", dpi=600)
    else:
        plt.savefig("Tab_B_Weight_list_"+filename+"_SW_"+str(sequence_window)+".pdf",dpi=600)
        plt.savefig("Tab_B_Weight_list_"+filename+"_SW_"+str(sequence_window)+".png",dpi=600)
#Model("Keras")
if __name__=='__main__':
    # Parameters
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv,wave_type, is_add_noise, noise_ratio, pooling_type, corss_val_label
    # ---------------------------Fixed Parameters------------------------------
    filepath = os.getcwd()

    sequence_window_list = [20]
    hidden_units = 200
    input_dim = 33
    number_class = 2
    cross_cv = 2
    fixed_seed_num = 1337
    is_add_noise = False
    noise_ratio = 0
    # -------------------------------------------------------------------------
    filename_list = ['HB_AS_Leak.txt']
    #filename_list = ["HB_AS_Leak.txt", "HB_Nimda.txt", "HB_Slammer.txt", "HB_Code_Red_I.txt"]

    corss_val_label_list = [0,1]

    learning_rate = 0.0001

    epoch = 5
    case_list = [2]

    wave_type = 'db1'
    pooling_type = 'max pooling'

    #case_list = [0]


    for filename in filename_list:
        if 'Nimda' in filename:multi_scale_value = 12
        else:multi_scale_value = 10

        for tab in range(len(corss_val_label_list)):
            corss_val_label = corss_val_label_list[tab]

            for sequence_window in sequence_window_list:

                train_loss_list = [[] for i in range(3)]
                train_acc_list = [[] for i in range(3)]
                val_loss_list = [[] for i in range(3)]
                val_acc_list = [[] for i in range(3)]
                weight_list = []
                for each_case in case_list:
                    if each_case == 0:
                        is_multi_scale = False#Original LSTM
                        training_level = -1

                    elif each_case == 1:# Original + Attention LSTM
                        is_multi_scale = True

                        training_level = multi_scale_value

                    elif each_case == 2:#MS-LSTM
                        is_multi_scale = True
                        training_level = multi_scale_value

                    train_loss,val_loss,train_acc,val_acc,weight_list,RESULT = Model(each_case,"MS-LSTM")
                    if each_case == 1:
                        try:
                            weight_plotting(filename, weight_list)
                        except:
                            pass
                    #train_loss,val_loss,train_acc,val_acc = Model("Keras-LSTM")

                    #a,b = Model("Keras-LSTM")
                    #a,b = Model("Keras-RNN")

                    train_loss_list[each_case].extend(train_loss)
                    train_acc_list[each_case].extend(train_acc)
                    val_loss_list[each_case].extend(val_loss)
                    val_acc_list[each_case].extend(val_acc)

                #print(train_acc_list)
                epoch_loss_plotting(train_loss_list,val_loss_list)
                epoch_acc_plotting(train_acc_list,val_acc_list)
                #print(weight_list)
                print(filename+" is completed________________________with sequence window is "+str(sequence_window)+" and cross_val is "+str(corss_val_label))


