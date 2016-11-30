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
from sklearn import svm,datasets,preprocessing,linear_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from collections import defaultdict
import numpy as np
from numpy import *
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
#from sklearn import hmm
from sklearn import svm,datasets,preprocessing,linear_model
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,preprocessing,linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
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

def Manipulation(X_Taining,Y_Training,window_size):
    window_size = len(X_Taining[0])
    TEMP_XTraining = []
    N = window_size
    time_scale_size = 1
    for tab1 in range(len(X_Taining)):
        TEMP_XTraining.append([])
        for tab2 in range(N):
            TEMP_Value = np.zeros((1,len(X_Taining[0][0])))
            for tab3 in range(time_scale_size):
                TEMP_Value += X_Taining[tab1][tab2*time_scale_size+tab3]
            TEMP_Value = TEMP_Value/time_scale_size
            TEMP_XTraining[tab1].extend(list(TEMP_Value[0]))
    return TEMP_XTraining,Y_Training



def Model(Label):
    global filepath, filename, sequence_window, number_class, batch_size, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, multi_scale_value, training_level, cross_cv
    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY","F1_SCORE","AUC","G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []

    for tab_cv in range(cross_cv):

        if not tab_cv == 1 :continue
        epoch_training_loss_list = []
        epoch_val_loss_list = []
        print(is_multi_scale)
        x_train, y_train,y_train0,x_test, y_test, y_test0 = LoadData.GetData_WithoutS(filepath, filename, sequence_window,tab_cv,cross_cv,Multi_Scale=is_multi_scale,Wave_Let_Scale=training_level)

        #using MLP to train
        if Label == "MLP":
            start = time.clock()
            model = Sequential()
            model.add(Dense(200, activation="relu",input_dim=33))
            model.add(Dropout(0.5))
            model.add(Dense(output_dim=number_class))
            #model.add(Activation("sigmoid"))

            model.add(Activation("softmax"))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)

            #result = model.predict(X_Testing, batch_size=batch_size)
            result = model.predict(x_test)
            end = time.clock()
            print("The Time For LSTM is " + str(end - start))
            results = Evaluation.Evaluation(y_test, result)
            #print(result)
        elif Label == "SVM":
            y_train = y_train0
            y_test = y_test0
            clf = svm.SVC(kernel="rbf", gamma=0.0001, C=1000)
            #clf = MultinomialNB()
            print(y_train)
            clf.fit(x_train, y_train)
            result = clf.predict(x_test)
            results = Evaluation.Evaluation_WithoutS(y_test, result)
        elif Label == "NB":
            y_train = y_train0
            y_test = y_test0
            clf = MultinomialNB()
            clf.fit(x_train, y_train)
            result = clf.predict(x_test)
            results = Evaluation.Evaluation_WithoutS(y_test, result)
        elif Label == "SVM-S":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData(filepath,filename,sequence_window,tab_cv,cross_cv)
            print(x_train)
            x_train,y_train = Manipulation(x_train,y_train0,sequence_window)
            x_test, y_test = Manipulation(x_test, y_test0, sequence_window)
            clf = svm.SVC(kernel="rbf", gamma=0.0001, C=1000)
            clf.fit(x_train, y_train)
            result = clf.predict(x_test)
            results = Evaluation.Evaluation_WithoutS(y_test, result)
        elif Label == "NB-S":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData(filepath,filename,sequence_window,tab_cv,cross_cv)
            print(x_train)
            x_train,y_train = Manipulation(x_train,y_train0,sequence_window)
            x_test, y_test = Manipulation(x_test, y_test0, sequence_window)
            clf = MultinomialNB()
            clf.fit(x_train, y_train)
            result = clf.predict(x_test)
            results = Evaluation.Evaluation_WithoutS(y_test, result)

        print(results)
        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)


    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    print(result_list_dict)
    with open(os.path.join(os.getcwd(),"Comparison_Log"+filename+".txt"),"a")as fout:
        outfileline = Label+":__"
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
    global filepath, filename, sequence_window, number_class, batch_size, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, multi_scale_value, training_level, cross_cv
    # ---------------------------Fixed Parameters------------------------------
    filepath = os.getcwd()
    sequence_window = 32
    hidden_units = 200
    input_dim = 33
    number_class = 2
    cross_cv = 2
    fixed_seed_num = 1337
    # -------------------------------------------------------------------------
    filename = 'HB_Slammer.txt'
    learning_rate = 0.0001
    epoch = 100
    case_list = [1]
    multi_scale_value = 5
    batch_size = 1888
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

        a,b = Model("SVM-S")
        train_loss_list[each_case].extend(a)
        val_loss_list[each_case].extend(b)

    #epoch_loss_plotting(train_loss_list,val_loss_list)