import time
import loaddata
import evaluation
from sklearn.feature_selection import RFE
from collections import defaultdict
import numpy as np
from numpy import *
from sklearn import tree
from keras.models import Sequential
from keras.layers.core import, Activation
from keras.layers import Dense
from keras.models import Model
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
def Basemodel(label,para=[]):
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv, is_add_noise, noise_ratio
    try:
        filepath = para["filepath"]
        filename = para["filename"]
        sequence_window = para["sequence_window"]
        number_class = para["number_class"]
        hidden_units = para["hidden_units"]
        input_dim = para["input_dim"]
        learning_rate = para["learning_rate"]
        epoch = para["epoch"]
        is_multi_scale = para["is_multi_scale"]
        training_level = para["training_level"]
        cross_cv = para["cross_cv"]
        fixed_seed_num = para["fixed_seed_num"]
        is_add_noise = para["is_add_noise"]
        noise_ratio = para["noise_ratio"]
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
        #using MLP to train
        if label == "SVM":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(label+" is running..............................................")
            y_train = y_train0
            clf = svm.SVC(kernel="rbf", gamma=0.00001, C=100000,probability=True)
            print(x_train.shape)
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)
            return evaluation.evaluation(y_test, result)
            #results = evaluation.evaluation(y_test, result)

        elif label == "SVMF":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=5)

            print(label+" is running..............................................")
            clf = svm.SVC(kernel="rbf", gamma=0.00001, C=100000,probability=True)
            print(x_train.shape)
            #x_train_new = SelectKBest(f_classif, k=num_selected_features).fit_transform(x_train, y_train0)
            #x_test_new = SelectKBest(f_classif, k=num_selected_features).fit_transform(x_test, y_test0)

            clf.fit(x_train, y_train0)
            result = clf.predict_proba(x_test)
            #return evaluation.evaluation(y_test, result)
            #results = evaluation.evaluation(y_test, result)
        elif label == "SVMW":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=6)

            print(label + " is running..............................................")
            #SVR(kernel="linear") = svm.SVC(kernel="rbf", gamma=0.00001, C=100000, probability=True)
            estimator = svm.SVC(kernel="linear",probability=True)
            selector = RFE(estimator, num_selected_features, step=1)
            selector = selector.fit(x_train, y_train0)

            result = selector.predict_proba(x_test)
            # return evaluation.evaluation(y_test, result)
            # results = evaluation.evaluation(y_test, result)
        elif label == "NBF":

            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=10)

            print(label + " is running..............................................")
            clf = MultinomialNB()
            clf.fit(x_train, y_train0)
            result = clf.predict_proba(x_test)


        elif label == "NBW":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=11)

            print(label + " is running..............................................")
            #SVR(kernel="linear") = svm.SVC(kernel="rbf", gamma=0.00001, C=100000, probability=True)
            estimator = MultinomialNB()
            selector = RFE(estimator, num_selected_features, step=1)
            selector = selector.fit(x_train, y_train0)

            result = selector.predict_proba(x_test)
            # return evaluation.evaluation(y_test, result)
            # results = evaluation.evaluation(y_test, result)
        elif label == "NB":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=1)

            print(label+" is running..............................................")
            y_train = y_train0
            clf = MultinomialNB()
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

            #return evaluation.evaluation(y_test, result)
            #results = evaluation.evaluation(y_test, result)

        elif label == "DT":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=2)

            print(label+" is running.............................................."+str(x_train.shape))
            y_train = y_train0
            clf = tree.DecisionTreeClassifier()
            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

            #return evaluation.evaluation(y_test, result)
            #results = evaluation.evaluation(y_test, result)
        elif label == "Ada.Boost":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(label+" is running.............................................."+str(x_train.shape))
            y_train = y_train0
            #clf = AdaBoostClassifier(n_estimators=10) #Nimda tab=1
            clf = AdaBoostClassifier(n_estimators=10)

            clf.fit(x_train, y_train)
            result = clf.predict_proba(x_test)

            #return evaluation.evaluation(y_test, result)
            #results = evaluation.evaluation(y_test, result)
        elif label == "MLP":
            x_train, y_train, y_train0, x_test, y_test, y_test0 = loaddata.GetData_WithoutS(is_add_noise,noise_ratio,filepath, filename,
                                                                                            sequence_window, tab_cv,
                                                                                            cross_cv,
                                                                                            Multi_Scale=is_multi_scale,
                                                                                            Wave_Let_Scale=training_level,
                                                                                            Normalize=0)

            print(label+" is running..............................................")
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

        elif label == "RNN":
            print(label+" is running..............................................")
            start = time.clock()
            x_train_multi_list, x_train, y_train, x_testing_multi_list, x_test, y_test = loaddata.GetData(is_add_noise,noise_ratio,'Attention',
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
        elif label == "LSTM":
            print(label+" is running..............................................")
            start = time.clock()
            x_train_multi_list, x_train, y_train, x_testing_multi_list, x_test, y_test = loaddata.GetData(is_add_noise,noise_ratio,'Attention',filepath,
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

        if len(para) > 0:
            return evaluation.evaluation(y_test, result)#Plotting AUC

        results = evaluation.evaluation(y_test, result)# Computing ACCURACY,F1-score,..,etc
        print(results)
        y_test2 = np.array(evaluation.ReverseEncoder(y_test))
        result2 = np.array(evaluation.ReverseEncoder(result))
        print("---------------------------1111111111111111")
        with open("StatFalseAlarm_"+filename+"_True.txt","w") as fout:
            for tab in range(len(y_test2)):
                fout.write(str(int(y_test2[tab]))+'\n')
        with open("StatFalseAlarm_"+filename+"_"+label+"_"+"_Predict.txt","w") as fout:
            for tab in range(len(result2)):
                fout.write(str(int(result2[tab]))+'\n')
        print(result2.shape)
        print("---------------------------22222222222222222")

        for each_eval, each_result in results.items():
            result_list_dict[each_eval].append(each_result)

    for eachk, eachv in result_list_dict.items():
        result_list_dict[eachk] = np.average(eachv)
    if is_add_noise == False:
        with open(os.path.join(os.getcwd(),"Comparison_Log_"+filename+".txt"),"a")as fout:
            outfileline = label+":__"
            fout.write(outfileline)
            for eachk,eachv in result_list_dict.items():
                fout.write(eachk+": "+str(round(eachv,3))+",\t")
            fout.write('\n')
    else:
        with open(os.path.join(os.getcwd(),"Comparison_Log_Adding_Noise_"+filename+".txt"),"a")as fout:
            outfileline = label+":__"+"Noise_Ratio_:"+str(noise_ratio)
            fout.write(outfileline)
            for eachk,eachv in result_list_dict.items():
                fout.write(eachk+": "+str(round(eachv,3))+",\t")
            fout.write('\n')

    return results
    #return epoch_training_loss_list,epoch_val_loss_list

def epoch_loss_plotting(train_loss_list,val_loss_list):
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

if __name__=='__main__':
    # para
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv, is_add_noise,noise_ratio
    # ---------------------------Fixed para------------------------------
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
    filename_list = ["HB_AS_Leak.txt", "HB_Nimda.txt", "HB_Slammer.txt", "HB_Code_Red_I.txt"]
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