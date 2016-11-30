import os
import time
import numpy as np
#import tflearn
import LoadData
import Evaluation
import tensorflow as tf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import RFE
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
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
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



def Model(Label,Parameters=[]):
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv, is_add_noise, noise_ratio
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
        is_add_noise = Parameters["is_add_noise"]
        noise_ratio = Parameters["noise_ratio"]
    except:
        pass


    result_list_dict = defaultdict(list)
    evaluation_list = ["ACCURACY","F1_SCORE","AUC","G_MEAN"]
    for each in evaluation_list:
        result_list_dict[each] = []
    np.random.seed(fixed_seed_num)  # for reproducibility
    #num_selected_features = 30
    #num_selected_features = 25#AS leak tab=0
    #num_selected_features = 32#Slammer tab=0
    num_selected_features = 33#Nimda tab=1
    for tab_cv in range(cross_cv):

        if not tab_cv == 0 :continue
        epoch_training_loss_list = []
        epoch_val_loss_list = []
        #print(is_multi_scale)

        #using MLP to train
        if Label == "SVM":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(Label+" is running..............................................")
            y_train = y_train0
            clf = svm.SVC(kernel="rbf", gamma=0.00001, C=100000,probability=True)
            print(x_train.shape)
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)
            #return Evaluation.Evaluation(y_test, result)
            #results = Evaluation.Evaluation(y_test, result)

        elif Label == "SVMF":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=5)

            print(Label+" is running..............................................")
            clf = svm.SVC(kernel="rbf", gamma=0.00001, C=100000,probability=True)
            print(x_train.shape)
            #x_train_new = SelectKBest(f_classif, k=num_selected_features).fit_transform(x_train, y_train0)
            #x_test_new = SelectKBest(f_classif, k=num_selected_features).fit_transform(x_test, y_test0)

            clf.fit(x_train, y_train0)
            result = clf.predict_proba(x_test)
            #return Evaluation.Evaluation(y_test, result)
            #results = Evaluation.Evaluation(y_test, result)
        elif Label == "SVMW":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=6)

            print(Label + " is running..............................................")
            #SVR(kernel="linear") = svm.SVC(kernel="rbf", gamma=0.00001, C=100000, probability=True)
            estimator = svm.SVC(kernel="linear",probability=True)
            selector = RFE(estimator, num_selected_features, step=1)
            selector = selector.fit(x_train, y_train0)

            result = selector.predict_proba(x_test)
            # return Evaluation.Evaluation(y_test, result)
            # results = Evaluation.Evaluation(y_test, result)
        elif Label == "NBF":

            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=10)

            print(Label + " is running..............................................")
            clf = MultinomialNB()
            clf.fit(x_train, y_train0)
            result = clf.predict_proba(x_test)


        elif Label == "NBW":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=11)

            print(Label + " is running..............................................")
            #SVR(kernel="linear") = svm.SVC(kernel="rbf", gamma=0.00001, C=100000, probability=True)
            estimator = MultinomialNB()
            selector = RFE(estimator, num_selected_features, step=1)
            selector = selector.fit(x_train, y_train0)

            result = selector.predict_proba(x_test)
            # return Evaluation.Evaluation(y_test, result)
            # results = Evaluation.Evaluation(y_test, result)
        elif Label == "NB":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=1)

            print(Label+" is running..............................................")
            y_train = y_train0
            clf = MultinomialNB()
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

            #return Evaluation.Evaluation(y_test, result)
            #results = Evaluation.Evaluation(y_test, result)

        elif Label == "DT":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=2)

            print(Label+" is running.............................................."+str(x_train.shape))
            y_train = y_train0
            clf = tree.DecisionTreeClassifier()
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

            #return Evaluation.Evaluation(y_test, result)
            #results = Evaluation.Evaluation(y_test, result)
        elif Label == "Ada.Boost":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(Label+" is running.............................................."+str(x_train.shape))
            y_train = y_train0
            #clf = AdaBoostClassifier(n_estimators=10) #Nimda tab=1
            clf = AdaBoostClassifier(n_estimators=10)

            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

            #return Evaluation.Evaluation(y_test, result)
            #results = Evaluation.Evaluation(y_test, result)
        elif Label == "MLP":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(Label+" is running..............................................")
            batch_size = len(y_train)
            start = time.clock()
            model = Sequential()
            model.add(Dense(hidden_units, activation="relu", input_dim=33))

            model.add(Dense(output_dim=number_class))
            model.add(Activation("sigmoid"))
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)
            #result = model.predict(X_Testing, batch_size=batch_size)
            result = model.predict(x_test)
            end = time.clock()
            print("The Time For MLP is " + str(end - start))

            #return Evaluation.Evaluation(y_test, result)
            #results = Evaluation.Evaluation(y_test, result)

        #elif Label == "SVM-S":
            #x_train, y_train, y_train0, x_test, y_test, y_test0 = LoadData.GetData('Attention',filepath,filename,sequence_window,tab_cv,cross_cv)
            #x_train,y_train = Manipulation(x_train,y_train0,sequence_window)
            #x_test, y_test = Manipulation(x_test, y_test0, sequence_window)
            #clf = svm.SVC(kernel="rbf")
            #clf.fit(x_train, y_train)
            #result = clf.predict(x_test)
            #results = Evaluation.Evaluation_WithoutS(y_test, result)
        elif Label == "RNN":
            print(Label+" is running..............................................")
            start = time.clock()
            x_train_multi_list, x_train, y_train, x_testing_multi_list, x_test, y_test = LoadData.GetData(is_add_noise,noise_ratio,'Attention',
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
            #model.add(Dense(30, activation="relu"))
            #model.add(Dropout(0.2))
            model.add(Dense(30, activation="sigmoid"))
            #model.add(Dropout(0.3))
            # model.add(Dense(5,activation="tanh"))

            model.add(Dense(output_dim=number_class))
            model.add(Activation("sigmoid"))
            # model.add(Activation("softmax"))

            # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)

            #result = model.predict(X_Testing, batch_size=batch_size)

            result = model.predict(x_test)

            #return Evaluation.Evaluation(y_test, result)
            #results = Evaluation.Evaluation(y_test, result)

            end = time.clock()
            print("The Time For RNN is " + str(end - start))

            # print(result)
        elif Label == "LSTM":
            print(Label+" is running..............................................")
            start = time.clock()
            x_train_multi_list, x_train, y_train, x_testing_multi_list, x_test, y_test = LoadData.GetData(is_add_noise,noise_ratio,'Attention',filepath,
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
            # model.add(LSTM(lstm_size,return_sequences=True,input_shape=(len(X_Training[0]),33)))
            # model.add(LSTM(100,return_sequences=True))
            # model.add(Dense(10, activation="tanh"))
            # model.add(Dense(5,activation="tanh"))
            model.add(Dense(30, activation="relu"))
            #model.add(Dropout(0.2))

            #model.add(Dense(30, activation="sigmoid"))
            #model.add(Dropout(0.3))
            # model.add(Dense(5,activation="tanh"))

            model.add(Dense(output_dim=number_class))
            model.add(Activation("sigmoid"))
            #model.add(Activation("softmax"))

            # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)

            #result = model.predict(X_Testing, batch_size=batch_size)

            result = model.predict(x_test)

            end = time.clock()
            print("The Time For LSTM is " + str(end - start))

        if len(Parameters) > 0:
            return Evaluation.Evaluation(y_test, result)#Plotting AUC

        results = Evaluation.Evaluation(y_test, result)# Computing ACCURACY,F1-score,..,etc
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
    #print(result_list_dict)
    if is_add_noise == False:
        with open(os.path.join(os.getcwd(),"Comparison_Log_"+filename+".txt"),"a")as fout:
            outfileline = Label+":__"
            fout.write(outfileline)
            for eachk,eachv in result_list_dict.items():
                fout.write(eachk+": "+str(round(eachv,3))+",\t")
            fout.write('\n')
    else:
        with open(os.path.join(os.getcwd(),"Comparison_Log_Adding_Noise_"+filename+".txt"),"a")as fout:
            outfileline = Label+":__"+"Noise_Ratio_:"+str(noise_ratio)
            fout.write(outfileline)
            for eachk,eachv in result_list_dict.items():
                fout.write(eachk+": "+str(round(eachv,3))+",\t")
            fout.write('\n')

    return results
    #return epoch_training_loss_list,epoch_val_loss_list

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
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv, is_add_noise,noise_ratio
    # ---------------------------Fixed Parameters------------------------------
    filepath = os.getcwd()#1
    sequence_window = 10#2
    hidden_units = 200#3
    input_dim = 33#4
    number_class = 2#5
    cross_cv = 2#6
    fixed_seed_num = 1337#7
    is_add_noise = False#8
    noise_ratio = 0#9
    # -------------------------------------------------------------------------
    learning_rate = 0.001#10
    epoch = 100#11
    training_level = -1#12
    is_multi_scale = False#13
    #filename_list = ["HB_AS_Leak.txt", "HB_Nimda.txt", "HB_Slammer.txt", "HB_Code_Red_I.txt"]
    filename_list = ["B_C_N_S_Multi.txt"]

    #learning_rate_list = [0.001, 0.0001, 0.001, 0.001]
    #filename = 'HB_Nimda.txt'


    #method_list = ["Ada.Boost"]
    #method_list = ["NBW"]
    method_list = ["SVM"]

    #method_list = ["SVM","SVMF","SVMW","NB","NBF","NBW","DT","Ada.Boost","MLP","RNN","LSTM"]

    for filename in filename_list:
        result_list = []
        for tab in range(len(method_list)):
            result_list.append(Model(method_list[tab]))

        for tab in range(len(method_list)):
            print(method_list[tab])
            print(result_list[tab])
        print(filename+" is completed!!!")

        #train_loss_list[each_case].extend(a)
        #val_loss_list[each_case].extend(b)

    #this is a a test
    #epoch_loss_plotting(train_loss_list,val_loss_list)