#_author_by_MC@20160424
import os
import time
start = time.time()
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
def get_auc(arr_score, arr_label, pos_label):
    score_label_list = []
    for index in xrange(len(arr_score)):
        score_label_list.append((float(arr_score[index]), int(arr_label[index])))
    score_label_list_sorted = sorted(score_label_list, key = lambda line:line[0], reverse = True)

    fp, tp = 0, 0
    lastfp, lasttp = 0, 0
    A = 0
    lastscore = None

    for score_label in score_label_list_sorted:
        score, label = score_label[:2]
        if score != lastscore:
            A += trapezoid_area(fp, lastfp, tp, lasttp)
            lastscore = score
            lastfp, lasttp = fp, tp
        if label == pos_label:
            tp += 1
        else:
            fp += 1

    A += trapezoid_area(fp, lastfp, tp, lasttp)
    A /= (fp * tp)
    return A
def trapezoid_area(x1, x2, y1, y2):
    delta = abs(x2 - x1)
    return delta * 0.5 * (y1 + y2)
def LoadData(input_data_path,filename):
    with open(os.path.join(input_data_path,filename)) as fin:
        global negative_sign,positive_sign
        if filename == 'sonar.dat':
            negative_flag = 'M'
        else:
            negative_flag = '1.0'
        Data=[]

        for each in fin:
            if '@' in each:continue
            val=each.split(",")
            if len(val)>0 or val[-1].strip()=="negative" or val[-1].strip()=="positive":
                #print(each)
                if val[-1].strip()== negative_flag:
                    val[-1] = negative_sign
                else:
                    val[-1] = positive_sign
                try:
                    val=map(lambda a:float(a),val)
                except:
                    val=map(lambda a:str(a),val)

                val[-1]=int(val[-1])
                Data.append(val)
        Data=np.array(Data)
        return Data

def Compute_average_list(mylist):
    temp = 0
    for i in range(len(mylist)):
        temp += float(mylist[i])
    return float(temp)/len(mylist)

def reConstruction(window_size,data,label):
    newdata = []
    newlabel = []
    L = len(data)
    D = len(data[0])
    interval = 1

    index = 0
    newdata_count = 0
    initial_value = -999
    while index+window_size < L:
        newdata.append(initial_value)
        newlabel.append(initial_value)
        Sequence = []
        for i in range(window_size):
            Sequence.append(data[index+i])
            newlabel[newdata_count] = label[index+i]
        index += interval
        newdata[newdata_count]=Sequence
        newdata_count += 1
    return np.array(newdata),np.array(newlabel)



def Manipulation(X_Taining,Y_Training,time_scale_size):
    window_size = len(X_Taining[0])
    TEMP_XTraining = []
    N = window_size/time_scale_size
    for tab1 in range(len(X_Taining)):
        TEMP_XTraining.append([])
        for tab2 in range(N):
            TEMP_Value = np.zeros((1,len(X_Taining[0][0])))
            for tab3 in range(time_scale_size):
                TEMP_Value += X_Taining[tab1][tab2*time_scale_size+tab3]
            TEMP_Value = TEMP_Value/time_scale_size
            TEMP_XTraining[tab1].extend(list(TEMP_Value[0]))
    return TEMP_XTraining,Y_Training
def returnPositiveIndex(Data,negative_sign):
    temp = []
    for i in range(len(Data)):
        if int(Data[i][-1]) != negative_sign:
            temp.append(i)
    return temp

def returnNegativeIndex(Data,negative_sign):
    temp = []
    for i in range(len(Data)):
        if Data[i][-1] == negative_sign:
            temp.append(i)
    return temp

def MyEvaluation(Y_Testing, Result):
    acc = 0
    if len(Y_Testing)!=len(Result):
        print("Error!!!")
    else:
        for tab1 in range(len(Result)):
            temp_result = map(lambda a:int(round(a)),Result[tab1])
            temp_true = map(lambda a:int(round(a)),Y_Testing[tab1])
            if list(temp_result) == list(temp_true):
                acc += 1
    return round(float(acc)/len(Result),3)

def Convergge_BBB(X_Training,Y_Training,components):
    window_size = len(X_Training[0])
    #pca = RandomizedPCA(n_components=components)
    #lda = LinearDiscriminantAnalysis(n_components=2)
    #svd = TruncatedSVD(n_components=components)
    #X_r2 = lda.fit(X, y).transform(X)
    components2 =3
    pca = PCA(n_components=components2)
    TEMP_XData = []
    import pywt
    for tab1 in range(len(X_Training)):
        TEMP_XData.append([])
        X = np.transpose(X_Training[tab1])

        #coeffs = pywt.dwt(X, 'db3')

        coeffs = pywt.wavedec(X, 'db1', level=4)

        X_CA = coeffs[0]
        X_CD1 = coeffs[1]
        X_CD2 = coeffs[2]
        X_CD3 = coeffs[3]
        X_CD4 = coeffs[4]

        #X = np.concatenate((X1,X2),axis=1)
        #y = [Y_Training[tab1] for i in range(len(X_Training[tab1]))]
        #X = lda.fit(X,y).transform(X)
        #X = np.transpose(X)
        #print(X.shape)
        #print(ddd)

        #pca.fit(X_CD1)
        #X_WAVELET_PCA = pca.transform(X_CD1)
        #X_CA_PCA_REC = pca.inverse_transform(X_WAVELET_PCA)
        #coeffs[1] = X_CA_PCA_REC

        #pca.fit(X_CD2)
        #X_WAVELET_PCA = pca.transform(X_CD2)
        #X_CA_PCA_REC = pca.inverse_transform(X_WAVELET_PCA)
        #coeffs[2] = X_CA_PCA_REC

        #pca.fit(X_CD3)
        #X_WAVELET_PCA = pca.transform(X_CD3)
        #X_CA_PCA_REC = pca.inverse_transform(X_WAVELET_PCA)
        #coeffs[3] = X_CA_PCA_REC

        #pca.fit(X_CD4)
        #X_WAVELET_PCA = pca.transform(X_CD4)
        #X_CA_PCA_REC = pca.inverse_transform(X_WAVELET_PCA)
        #coeffs[4] = X_CA_PCA_REC

        coeffs[2] = None
        coeffs[3] = None
        coeffs[4] = None

        X_WAVELET_REC = pywt.waverec(coeffs, 'db1')
        #X_WAVELET_REC = pywt.idwt(X_CA_PCA_REC,X_CD, 'db1')

        X_1 = np.transpose(X_WAVELET_REC)
        #


        #print(X_1.shape)


        #X = svd.fit_transform(X)
        TEMP_XData[tab1].extend(X_1)
    #print("111111111111111111111111111111111111111111111111111111111111111")
    #print(np.array(TEMP_XData)[0])
    #print("999999999999999999999999999999999999999999999999999999999999999")
    #print(fs)
    return  np.array(TEMP_XData),Y_Training


def Convergge(X_Training,Y_Training,time_scale_size):
    window_size = len(X_Training[0])
    TEMP_XData = []
    N = window_size/time_scale_size
    for tab1 in range(len(X_Training)):
        TEMP_XData.append([])
        for tab2 in range(N):
            TEMP_Value = np.zeros((1,len(X_Training[0][0])))
            for tab3 in range(time_scale_size):
                TEMP_Value += X_Training[tab1][tab2*time_scale_size+tab3]
            TEMP_Value = TEMP_Value/time_scale_size
            TEMP_XData[tab1].append(list(TEMP_Value[0]))
    #print("111111111111111111111111111111111111111111111111111111111111111")
    #print(np.array(TEMP_XData)[0])
    #print("999999999999999999999999999999999999999999999999999999999999999")
    #print(fs)

    return  np.array(TEMP_XData),Y_Training
def Add_Noise(Ratio,Data):
    w = 5
    X = Data[:,:-1]
    Y = Data[:,-1]
    Std_List = X.std(axis=0,ddof=0)
    N = int(Ratio*len(Data))
    Noise = []
    for tab1 in range(N):
        Base_Instance_Index = random.randint(0,len(Data)-1)
        Base_Instance = Data[Base_Instance_Index]
        Noise.append([])
        for tab2 in range(len(Std_List)):
            temp = random.uniform(Std_List[tab2]*-1,Std_List[tab2])
            Noise[tab1].append(float(Base_Instance[tab2]+temp/w))
        Noise[tab1].append(Base_Instance[-1])
    Noise = np.array(Noise)
    return np.concatenate((Data,Noise),axis=0)
def Add_Vector(Matrix):
    N = len(Matrix)
    D = len(Matrix[0])
    sum = [0 for i in range(D)]
    for tab1 in range(N):
        for tab2 in range(D):
            sum[tab2] = sum[tab2] + Matrix[tab1][tab2]
    average = map(lambda a:float(a)/N,sum)
    return average
def MeanPooling(Data):
    N = len(Data)
    Temp = []
    for tab1 in range(N):
        A = Data[tab1]
        Temp.append(Add_Vector(A))
    return np.array(Temp)

def Main(Bin_or_Multi_Label,Method_Dict,filename,window_size_label,lstm_size,Noise_Ratio,window_size=0,time_scale_size=0):

    global ACCURACY,positive_sign,negative_sign,input_data_path,output_data_path,Normalization_Method
    if Normalization_Method == "_Std":
        scaler = preprocessing.StandardScaler()
    elif Normalization_Method == "_L2norm":
        scaler = preprocessing.Normalizer()
    elif Normalization_Method == "_Minmax":
        scaler = preprocessing.MinMaxScaler()

    Data_=LoadData(input_data_path,filename)
    cross_folder = 2
    #Data_ = Add_Noise(Noise_Ratio,Data_)
    #Pos_Data=Data_[Data_[:,-1]!=negative_sign]
    #Neg_Data=Data_[Data_[:,-1]==negative_sign]
    PositiveIndex = returnPositiveIndex(Data_,negative_sign)
    NegativeIndex = returnNegativeIndex(Data_,negative_sign)

    if Bin_or_Multi_Label=="Multi":np.random.shuffle(PositiveIndex)
    Pos_Data = Data_[PositiveIndex]
    Neg_Data = Data_[NegativeIndex]

    #print(Data_[PositiveIndex,-1])

    #print(Data_[NegativeIndex,-1])

    Auc_list = {}
    ACC_R_list = {}
    ACC_A_list = {}
    G_mean_list = {}
    ACC_list = {}
    F1_list = {}

    Deviation_ACC_R_list={}
    Deviation_ACC_A_list={}
    Deviation_Auc_list={}
    Deviation_G_mean_list={}
    Deviation_ACC_list = {}
    Deviation_F1_list = {}

    Temp_SubFeature_Auc_list = {}
    Temp_SubFeature_G_mean_list = {}
    Temp_SubFeature_ACC_R_list = {}
    Temp_SubFeature_ACC_A_list = {}
    Temp_SubFeature_ACC_list = {}
    Temp_SubFeature_F1_list = {}
    for eachMethod,methodLabel in Method_Dict.items():
        Auc_list[eachMethod] = []
        ACC_R_list[eachMethod] = []
        ACC_A_list[eachMethod] = []
        G_mean_list[eachMethod] = []
        ACC_list[eachMethod] = []
        F1_list[eachMethod] = []
    for eachMethod,methodLabel in Method_Dict.items():
        print(eachMethod+" is running------------->>>>>>>>>>>>>>>>>>>>>")
        if eachMethod=='NB':scaler = preprocessing.MinMaxScaler()

        #print(eachMethod+" is running...")
        Top_K_List = []
        Total_Dimensions = len(Data_[0])-1

        Iterations = 1
        for Top_K in range(Total_Dimensions,Total_Dimensions+1,2):
            #D=[1/float(len(Y_Training)) for i in range(len(Y_Training))]
            #Sub_Features=sorted(Return_Top_K_Features(Training_Data[:,:-1],Y_Training,D,Top_K))
            Temp_SubFeature_ACC_R_list[eachMethod+"_TF_"+str(Top_K)] = []
            Temp_SubFeature_ACC_A_list[eachMethod+"_TF_"+str(Top_K)] = []
            Temp_SubFeature_Auc_list[eachMethod+"_TF_"+str(Top_K)] = []
            Temp_SubFeature_G_mean_list[eachMethod+"_TF_"+str(Top_K)] = []
            Temp_SubFeature_ACC_list[eachMethod+"_TF_"+str(Top_K)] = []
            Temp_SubFeature_F1_list[eachMethod+"_TF_"+str(Top_K)] = []


            Deviation_ACC_R_list[eachMethod+"_TF_"+str(Top_K)] =[]
            Deviation_ACC_A_list[eachMethod+"_TF_"+str(Top_K)] =[]
            Deviation_Auc_list[eachMethod+"_TF_"+str(Top_K)] =[]
            Deviation_G_mean_list[eachMethod+"_TF_"+str(Top_K)] =[]
            Deviation_ACC_list[eachMethod+"_TF_"+str(Top_K)] =[]
            Deviation_F1_list[eachMethod+"_TF_"+str(Top_K)] =[]

            #print("The Top_K is :"+str(Top_K))
            #Top_K_List.append(Top_K)

            for iteration_count in range(Iterations):
                #print(str(iteration_count+1)+"th iterations is running...")
                cross_folder_auc_list=[]
                cross_folder_acc_r_list=[]
                cross_folder_acc_a_list=[]
                cross_folder_g_mean_list=[]
                cross_folder_acc_list=[]
                cross_folder_f1_list=[]

                for tab_cross in range(cross_folder):
                    #if not tab_cross==0:continue
                    Positive_Data_Index_Training=[]
                    Positive_Data_Index_Testing=[]
                    Negative_Data_Index_Training=[]
                    Negative_Data_Index_Testing=[]

                    for tab_positive in range(len(PositiveIndex)):
                        if int((cross_folder-tab_cross-1)*len(Pos_Data)/cross_folder)<=tab_positive<int((cross_folder-tab_cross)*len(Pos_Data)/cross_folder):
                            Positive_Data_Index_Testing.append(PositiveIndex[tab_positive])
                        else:
                            Positive_Data_Index_Training.append(PositiveIndex[tab_positive])


                    for tab_negative in range(len(NegativeIndex)):
                        if int((cross_folder-tab_cross-1)*len(Neg_Data)/cross_folder)<=tab_negative<int((cross_folder-tab_cross)*len(Neg_Data)/cross_folder):
                            Negative_Data_Index_Testing.append(NegativeIndex[tab_negative])
                        else:
                            Negative_Data_Index_Training.append(NegativeIndex[tab_negative])

                    Training_Data_Index=np.append(Negative_Data_Index_Training,Positive_Data_Index_Training,axis=0)
                    Training_Data_Index.sort()

                    #print(Training_Data_Index)

                    Training_Data_Index = map(lambda a:int(a),Training_Data_Index)

                    #print(Training_Data_Index)

                    Training_Data = Data_[Training_Data_Index,:]
                    #print(list(Data_[Positive_Data_Index_Training,-1]).count(4))
                    #print(list(Pos_Data[:,-1]).count(4))

                    Testing_Data_Index=np.append(Negative_Data_Index_Testing,Positive_Data_Index_Testing,axis=0)

                    Testing_Data_Index.sort()

                    Testing_Data_Index = map(lambda a:int(a),Testing_Data_Index)

                    Testing_Data = Data_[Testing_Data_Index,:]

                    print(str(tab_cross+1)+"th Cross Validation is running and the training size is "+str(len(Training_Data))+", testing size is "+str(len(Testing_Data))+"......")

                    #positive_=Training_Data[Training_Data[:,-1]==positive_sign]
                    #negative_=Training_Data[Training_Data[:,-1]==negative_sign]
                    #print("IR is :"+str(float(len(negative_))/len(positive_)))
                    import pywt
                    if window_size_label == "true":
                        if methodLabel == 0:
                            np.random.seed(1337)  # for reproducibility
                            batch_size = 300
                            X_Testing_0 = scaler.fit_transform(Testing_Data[:, :-1])
                            Y_Testing_0 = Testing_Data[:,-1]
                            (X_Training_1, Y_Training_1) = reConstruction(window_size, scaler.fit_transform(Training_Data[:,:-1]),Training_Data[:,-1])
                            #X_Training, Y_Training = Convergge_BBB(X_Training_1, Y_Training_1, time_scale_size,40)
                            #X_Training, Y_Training = Convergge_BBB(X_Training_1, Y_Training_1, time_scale_size,20)
                            #X_Training, Y_Training = Convergge_BBB(X_Training, Y_Training, time_scale_size,10)
                            #X_Training, Y_Training = Convergge_BBB(X_Training, Y_Training, time_scale_size,5)
                            #print(len(Y_Training_0==0))
                            selected_features = 4


                            #N0 = X_Training[:,Y_Training==1]
                            if eachMethod == "LSTM":
                                X_Training, Y_Training = Convergge(X_Training_1, Y_Training_1, time_scale_size)
                            elif eachMethod == "MSLSTM":
                                X_Training, Y_Training = Convergge_BBB(X_Training_1, Y_Training_1, time_scale_size)





                            s = time.clock()
                            #(X_Testing_1, Y_Testing_1) = reConstruction(window_size, scaler.fit_transform(Testing_Data[:,:-1]),Testing_Data[:,-1])
                            (X_Testing_1, Y_Testing_1) = reConstruction(window_size, scaler.fit_transform(Testing_Data[:,:-1]),Testing_Data[:,-1])
                            #X_Testing, Y_Testing = Convergge_BBB(X_Testing_1, Y_Testing_1, time_scale_size,40)
                            #X_Testing, Y_Testing = Convergge_BBB(X_Testing_1, Y_Testing_1, time_scale_size,20)
                            #X_Testing, Y_Testing = Convergge_BBB(X_Testing, Y_Testing, time_scale_size,10)
                            #X_Testing, Y_Testing = Convergge_BBB(X_Testing, Y_Testing, time_scale_size,5)


                            """
                            P0 = X_Testing_0[Y_Testing_0 == 0, :][:, selected_features]
                            P0_Index = returnPositiveIndex(Training_Data, 1)
                            N0_Index = returnNegativeIndex(Training_Data, 1)
                            Y0 = X_Testing_0[:, selected_features]
                            X0 = [i for i in range(len(Y0))]
                            """

                            if eachMethod == "LSTM":
                                X_Testing, Y_Testing = Convergge(X_Testing_1, Y_Testing_1, time_scale_size)
                            elif eachMethod == "MSLSTM":
                                X_Testing, Y_Testing = Convergge_BBB(X_Testing_1, Y_Testing_1, time_scale_size)



                            """
                            AAA = MeanPooling(X_Testing[Y_Testing == 0, :])[:, selected_features]
                            BBB = MeanPooling(X_Testing[Y_Testing == 1, :])[:, selected_features]

                            P1 = X_Testing[Y_Testing == 0, :][:, 0][:, selected_features]
                            N1 = X_Testing[Y_Testing == 1, :][:, 0][:, selected_features]
                            plt.subplot(1, 2, 1)
                            plt.plot(X0, Y0, 'b')
                            plt.plot(P0_Index, P0, 'r')
                            plt.xlim(window_size, len(X0))
                            plt.ylim(-0.1, 0.4)
                            plt.subplot(1, 2, 2)
                            # plt.plot(X1[filter(lambda a:a<len(P0_Index),X1),:],Y1[filter(lambda a:a<len(P0_Index),X1),:],'b')
                            plt.plot(N0_Index[window_size - 1:], BBB, 'b')
                            P0_Index.pop(-1)
                            plt.xlim(window_size, len(X0))
                            plt.plot(P0_Index, AAA, 'r')
                            plt.ylim(-0.1, 0.4)
                            #plt.show()
                            print(AAA.shape)
                            print(BBB.shape)
                            """



                            d = time.clock()
                            print("For LSTM The Sequence and Time Scale is: "+str(d-s))
                            print(X_Training.shape)
                            print("Time Scale is "+str(time_scale_size))

                            #encoder1 = LabelEncoder()
                            #encoder1.fit(Y_Training)
                            #encoded_Y1 = encoder1.transform(Y_Training)
                            # convert integers to dummy variables (i.e. one hot encoded)
                            #Y_Training = np_utils.to_categorical(encoded_Y1)

                            #encoder2 = LabelEncoder()
                            #encoder2.fit(Y_Testing)
                            #encoded_Y2 = encoder2.transform(Y_Testing)
                            # convert integers to dummy variables (i.e. one hot encoded)
                            #Y_Testing = np_utils.to_categorical(encoded_Y2)


                            #print(X_Training.shape)
                            lstm_object = LSTM(200, input_length=len(X_Training[0]), input_dim=33)
                            #print('Build model...' + 'Window Size is ' + str(window_size) + ' LSTM Size is ' + str(
                                #lstm_size) + " Time Scale is " + str(time_scale_size))

                            #print(X_Training.shape)
                            #print(X_Training[0])
                            model = Sequential()

                            model.add(lstm_object)  # X.shape is (samples, timesteps, dimension)
                            #model.add(LSTM(lstm_size,return_sequences=True,input_shape=(len(X_Training[0]),33)))
                            #model.add(LSTM(100,return_sequences=True))
                            #model.add(LSTM(50))
                            model.add(Dense(10,activation="tanh"))
                            #model.add(Dense(5,activation="tanh"))

                            model.add(Dense(output_dim=1))
                            model.add(Activation("sigmoid"))
                            #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
                            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                            model.fit(X_Training, Y_Training, batch_size=batch_size, nb_epoch=20)

                            start = time.clock()
                            #result = model.predict(X_Testing, batch_size=batch_size)

                            result = model.predict(X_Testing)

                            end = time.clock()
                            print("The Time For LSTM is "+str(end-start))

                            del model


                        else:
                            (X_Training_1, Y_Training_1) = reConstruction(window_size, Training_Data[:,:-1],Training_Data[:,-1])
                            (X_Testing_1, Y_Testing_1) = reConstruction(window_size, Testing_Data[:,:-1], Testing_Data[:,-1])
                            X_Training,Y_Training = Manipulation(X_Training_1,Y_Training_1,time_scale_size)
                            X_Testing,Y_Testing = Manipulation(X_Testing_1,Y_Testing_1,time_scale_size)

                            X_Training = scaler.fit_transform(X_Training)
                            X_Testing = scaler.fit_transform(X_Testing)

                            print('Window Size is ' + str(window_size) + " Time Scale is " + str(time_scale_size)+"Tesing Length is :"+str(len(Y_Testing)))
                            if methodLabel == 1:
                                # clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                                clf = AdaBoostClassifier()
                            elif methodLabel == 2:
                                clf = tree.DecisionTreeClassifier()
                            elif methodLabel == 3:
                                clf = svm.SVC(kernel="rbf", gamma=0.001, C=1000)
                            elif methodLabel == 4:
                                clf = linear_model.LogisticRegression()
                            elif methodLabel == 5:
                                #clf = KNeighborsClassifier(3)
                                from sklearn.naive_bayes import GaussianNB
                                clf = GaussianNB()
                            clf.fit(X_Training, Y_Training)

                            start = time.clock()
                            result = clf.predict(X_Testing)
                            end = time.clock()
                            #print("The Time For "+eachMethod+" is "+str(end-start))

                    else:
                        X_Training = Training_Data[:,:-1]
                        Y_Training = Training_Data[:,-1]

                        X_Testing = Testing_Data[:,:-1]
                        Y_Testing = Testing_Data[:,-1]
                        X_Training = scaler.fit_transform(X_Training)

                        s = time.clock()
                        X_Testing = scaler.fit_transform(X_Testing)
                        print("Preprocessing For "+eachMethod+" is "+str(time.clock()-s))
                        if methodLabel == 1:
                            #clf = GradientBoostingClassifier(loss='deviance',n_estimators=300, learning_rate=0.1,max_depth=2)
                            clf = AdaBoostClassifier()
                        elif methodLabel == 2:
                            clf = tree.DecisionTreeClassifier()
                            #clf = hmm.GaussianHMM(3,"full")
                        elif methodLabel == 3:
                            clf = svm.SVC(kernel="rbf", gamma=0.0001, C=10000)
                        elif methodLabel == 4:
                            clf = linear_model.LogisticRegression()
                        elif methodLabel == 5:
                            #clf = KNeighborsClassifier(15)
                            clf = MultinomialNB()
                        #print(str(tab_cross+1)+"th cross validation is running and the training size is "+str(len(X_Training))+", testing size is "+str(len(X_Testing))+"......")

                        clf.fit(X_Training, Y_Training)
                        start = time.clock()
                        result = clf.predict(X_Testing)
                        end = time.clock()
                        print("The Time For "+eachMethod+" is "+str(end-start))
                    #positive_=Training_Data[Training_Data[:,-1]==positive_sign]
                    #negative_=Training_Data[Training_Data[:,-1]==negative_sign]
                    #print("IR is :"+str(float(len(negative_))/len(positive_)))

                    #try:
                        #if methodLabel == 0:
                            #ACC = MyEvaluation(Y_Testing, result)

                    print('Window Size is ' + str(window_size) + " Time Scale is " + str(
                        time_scale_size) + "Tesing Length is :" + str(len(Y_Testing)))

                    if 1>0:

                        Output=[]
                        if len(result)==len(Y_Testing):
                            for tab in range(len(Y_Testing)):
                                Output.append(int(round(result[tab])))
                        else:print("Error!")

                        ac_positive=0
                        ac_negative=0
                        for tab in range(len(Output)):
                            if Output[tab]==int(Y_Testing[tab]):
                                if Output[tab]==negative_sign:
                                    ac_negative += 1
                                else:
                                    ac_positive += 1
                        if tab_cross==0:

                            #print('Window Size is ' + str(window_size) + " Time Scale is " + str(time_scale_size)+"Tesing Length is :"+str(len(Y_Testing)))
                            with open("CrossTab_"+str(tab_cross)+"_"+str(eachMethod)+".txt","w")as fout:
                                for tab in range(len(Y_Testing)):
                                    fout.write(str(Output[tab])+'\t\t'+str(Y_Testing[tab])+'\n')

                        try:
                            ACC_R = float(ac_negative)/Output.count(negative_sign)
                        except:
                            ACC_R = float(ac_negative)*100/(Output.count(negative_sign)+1)
                        try:
                            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+str(ac_positive))
                            print(Output.count(-1))
                            ACC_A = float(ac_positive)/Output.count(0)
                        except:
                            ACC_A = float(ac_positive)*100/(Output.count(-1)+1)


                        #auc = roc_auc_score(Y_Testing,Output)
                        #g_mean=np.sqrt(float(ac_positive*ac_negative)/(len(np.array(Y_Testing)[np.array(Y_Testing)==positive_sign])*len(np.array(Y_Testing)[np.array(Y_Testing)==negative_sign])))

                        precision = ACC_A
                        recall = float(ac_positive)/list(Y_Testing).count(0)
                        ACC = round(float(ac_positive+ac_negative)/len(Output),5)
                        try:
                            f1_score = round((2*precision*recall)/(precision+recall),5)
                        except:
                            f1_score = 0.01*round((2*precision*recall)/(precision+recall+1),5)


                    cross_folder_acc_r_list.append(ACC_R*100)
                    cross_folder_acc_a_list.append(ACC_A*100)


                    #cross_folder_auc_list.append(auc*100)
                    #cross_folder_g_mean_list.append(g_mean*100)


                    cross_folder_acc_list.append(ACC*100)
                    cross_folder_f1_list.append(f1_score*100)

                for tab1 in range(int(len(cross_folder_acc_list)/cross_folder)):
                    temp_acc_r=0.0
                    temp_acc_a=0.0

                    #temp_auc=0.0
                    #temp_g_mean=0.0

                    temp_acc=0.0
                    temp_f1_score=0.0
                    for tab2 in range(cross_folder):
                        temp_acc_r += cross_folder_acc_r_list[tab1*cross_folder+tab2]
                        temp_acc_a += cross_folder_acc_a_list[tab1*cross_folder+tab2]

                        #temp_auc += cross_folder_auc_list[tab1*cross_folder+tab2]
                        #temp_g_mean += cross_folder_g_mean_list[tab1*cross_folder+tab2]

                        temp_acc += cross_folder_acc_list[tab1*cross_folder+tab2]
                        temp_f1_score += cross_folder_f1_list[tab1*cross_folder+tab2]

                    temp_acc_r=temp_acc_r/float(cross_folder)
                    temp_acc_a=temp_acc_a/float(cross_folder)

                    #temp_auc=temp_auc/float(cross_folder)
                    #temp_g_mean=temp_g_mean/float(cross_folder)

                    temp_acc=temp_acc/float(cross_folder)
                    temp_f1_score=temp_f1_score/float(cross_folder)

                    Temp_SubFeature_ACC_R_list[eachMethod+"_TF_"+str(Top_K)].append(temp_acc_r)
                    Temp_SubFeature_ACC_A_list[eachMethod+"_TF_"+str(Top_K)].append(temp_acc_a)

                    #Temp_SubFeature_Auc_list[eachMethod+"_TF_"+str(Top_K)].append(temp_auc)
                    #Temp_SubFeature_G_mean_list[eachMethod+"_TF_"+str(Top_K)].append(temp_g_mean)

                    Temp_SubFeature_ACC_list[eachMethod+"_TF_"+str(Top_K)].append(temp_acc)
                    Temp_SubFeature_F1_list[eachMethod+"_TF_"+str(Top_K)].append(temp_f1_score)


            deviation_acc_r=0.0
            deviation_acc_a=0.0

            #deviation_auc=0.0
            #deviation_g_mean=0.0

            deviation_acc=0.0
            deviation_f1_score=0.0

            mean_acc_r=Compute_average_list(Temp_SubFeature_ACC_R_list[eachMethod+"_TF_"+str(Top_K)])
            mean_acc_a=Compute_average_list(Temp_SubFeature_ACC_A_list[eachMethod+"_TF_"+str(Top_K)])

            #mean_auc=Compute_average_list(Temp_SubFeature_Auc_list[eachMethod+"_TF_"+str(Top_K)])
            #mean_g_mean=Compute_average_list(Temp_SubFeature_G_mean_list[eachMethod+"_TF_"+str(Top_K)])

            #print(Temp_SubFeature_ACC_list[eachMethod+"_TF_"+str(Top_K)])
            mean_acc=Compute_average_list(Temp_SubFeature_ACC_list[eachMethod+"_TF_"+str(Top_K)])
            mean_f1_score=Compute_average_list(Temp_SubFeature_F1_list[eachMethod+"_TF_"+str(Top_K)])

            for tab in range(len(Temp_SubFeature_Auc_list[eachMethod+"_TF_"+str(Top_K)])):
                temp_acc_r = Temp_SubFeature_ACC_R_list[eachMethod+"_TF_"+str(Top_K)][tab]
                temp_acc_a = Temp_SubFeature_ACC_A_list[eachMethod+"_TF_"+str(Top_K)][tab]

                deviation_acc_r=deviation_acc_r+((temp_acc_r-mean_acc_r)*(temp_acc_r-mean_acc_r))
                deviation_acc_a=deviation_acc_a+((temp_acc_a-mean_acc_a)*(temp_acc_a-mean_acc_a))

                #temp_auc = Temp_SubFeature_Auc_list[eachMethod+"_TF_"+str(Top_K)][tab]
                #deviation_auc=deviation_auc+((temp_auc-mean_auc)*(temp_auc-mean_auc))

                #temp_g_mean = Temp_SubFeature_G_mean_list[eachMethod+"_TF_"+str(Top_K)][tab]
                #deviation_g_mean=deviation_g_mean+((temp_g_mean-mean_g_mean)*(temp_g_mean-mean_g_mean))

                temp_acc = Temp_SubFeature_ACC_list[eachMethod+"_TF_"+str(Top_K)][tab]
                deviation_acc=deviation_acc+((temp_acc-mean_acc)*(temp_acc-mean_acc))

                #temp_f1_score = Temp_SubFeature_F1_list[eachMethod+"_TF_"+str(Top_K)][tab]
                #deviation_f1_score=deviation_f1_score+((temp_f1_score-mean_f1_score)*(temp_f1_score-mean_f1_score))

            deviation_acc_r/=Iterations
            deviation_acc_a/=Iterations

            #deviation_auc/=Iterations
            #deviation_g_mean/=Iterations

            deviation_acc/=Iterations
            deviation_f1_score/=Iterations
            Deviation_ACC_R_list[eachMethod+"_TF_"+str(Top_K)].append(deviation_acc_r)
            Deviation_ACC_A_list[eachMethod+"_TF_"+str(Top_K)].append(deviation_acc_a)

            #Deviation_Auc_list[eachMethod+"_TF_"+str(Top_K)].append(deviation_auc)
            #Deviation_G_mean_list[eachMethod+"_TF_"+str(Top_K)].append(deviation_g_mean)

            Deviation_ACC_list[eachMethod+"_TF_"+str(Top_K)].append(deviation_acc)
            Deviation_F1_list[eachMethod+"_TF_"+str(Top_K)].append(deviation_f1_score)



            ACC_R_list[eachMethod].append(Compute_average_list(Temp_SubFeature_ACC_R_list[eachMethod+"_TF_"+str(Top_K)]))
            ACC_A_list[eachMethod].append(Compute_average_list(Temp_SubFeature_ACC_A_list[eachMethod+"_TF_"+str(Top_K)]))
            ACC_list[eachMethod].append(Compute_average_list(Temp_SubFeature_ACC_list[eachMethod+"_TF_"+str(Top_K)]))

            #Auc_list[eachMethod].append(Compute_average_list(Temp_SubFeature_Auc_list[eachMethod+"_TF_"+str(Top_K)]))
            #G_mean_list[eachMethod].append(Compute_average_list(Temp_SubFeature_G_mean_list[eachMethod+"_TF_"+str(Top_K)]))

            F1_list[eachMethod].append(Compute_average_list(Temp_SubFeature_F1_list[eachMethod+"_TF_"+str(Top_K)]))



    Write_Out(output_data_path,filename,time_scale_size,ACC_R_list,"ACC_Regular")
    Write_Out(output_data_path,filename,time_scale_size,ACC_A_list,"ACC_Anomaly")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_ACC_R_list,"SubFeature_ACC_Regular",Deviation_ACC_R_list)
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_ACC_A_list,"SubFeature_ACC_Anomaly",Deviation_ACC_A_list)

    #Write_Out(output_data_path,filename,time_scale_size,Auc_list,"Auc")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_Auc_list,"SubFeature_Auc",Deviation_Auc_list)
    #Write_Out(output_data_path,filename,time_scale_size,G_mean_list,"G_mean")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_G_mean_list,"SubFeature_G_mean",Deviation_G_mean_list)

    Write_Out(output_data_path,filename,time_scale_size,ACC_list,"ACC")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_ACC_list,"SubFeature_ACC",Deviation_ACC_list)
    Write_Out(output_data_path,filename,time_scale_size,F1_list,"F1_score")
    #Write_Out(output_data_path,filename,time_scale_size,Temp_SubFeature_F1_list,"SubFeature_F1_score",Deviation_F1_list)
def Write_Out(filefolderpath,filename,time_scale_size,Result_List,Tag,Result_List_back=[]):
    with open(os.path.join(filefolderpath,filename.split('.')[0]+"_Info_"+Tag+"_List.txt"),"a")as fout:
        fout.write("-----------++++++########(time scale: "+str(time_scale_size)+ ")#######++++++-----------------\n")
        for eachk,eachv in Result_List.items():
            fout.write(eachk)
            fout.write(":\t\t")
            for each in eachv:
                fout.write("%.3f"%(each))
                fout.write("\t,")
            if len(Result_List_back) > 0:
                fout.write(str(Result_List_back[eachk]))
            fout.write('\n')

def get_all_subfactors(number):
    temp_list = []
    temp_list.append(1)
    temp_list.append(2)
    for i in range(3,number):
        if number%i == 0 :
            temp_list.append(i)
    temp_list.append(number)
    return temp_list

if __name__=='__main__':
    global ACCURACY,positive_sign,negative_sign,input_data_path,output_data_path,Normalization_Method
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign=0
    negative_sign=1
    ACCURACY=[]
    input_data_path =os.getcwd()
    Normalization_Method = "_L2norm"
    Bin_or_Multi_Label="Bin"
    #window_size_label_list = ['true','false']
    window_size_label_list = ['true']

    #window_size_list = [10,20,30,40,50,60]
    window_size_list = [32]
    filenamelist=os.listdir(input_data_path)
    lstm_size_list = [100]
    Noise_Ratio = 0.5
    for eachfile in filenamelist:
        if  not eachfile=='HB_Code_Red_I.txt':continue
        if '.py' in eachfile or '.DS_' in eachfile: continue
        if '.txt' in eachfile:
            pass
        else:
            continue
        print(eachfile+ " is processing---------------------------------------------------------------------------------------------LSTM_Size is "+str(lstm_size_list[-1]))
        for lstm_size in lstm_size_list:
            for window_size_label in window_size_label_list:
                if window_size_label == 'true':
                    #Method_Dict={"MSLSTM": 0}
                    #Method_Dict = {"LSTM": 0,"AdaBoost": 1, "DT": 2, "SVM": 3, "LOGREG": 4, "NB": 5}
                    Method_Dict = {"MSLSTM":0,"LSTM": 0,"AdaBoost": 1, "SVM": 3, "NB": 5}

                    for window_size in window_size_list:
                        time_scale_size_list = get_all_subfactors(window_size)
                        #output_data_path = os.path.join(os.getcwd(),'NNNNNBBBBB_'+str(Noise_Ratio*100)+'%'+'SSSingle_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        #output_data_path = os.path.join(os.getcwd(),'NNNNNBBBBB_'+'MMMulti_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        #output_data_path = os.path.join(os.getcwd(),'Noise_'+str(Noise_Ratio*100)+'%'+'SSSingle_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        output_data_path = os.path.join(os.getcwd(),'Half_Minutes_Window_Size_' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)

                        if not os.path.isdir(output_data_path):
                            os.makedirs(output_data_path)
                        #time_scale_size_list = [2,3]
                        for time_scale_size in time_scale_size_list:

                            if  time_scale_size > 1: continue
                            Main(Bin_or_Multi_Label,Method_Dict,eachfile,window_size_label,lstm_size,Noise_Ratio,window_size,time_scale_size)

                else:
                    #continue
                    Method_Dict={"NB": 5}

                    #Method_Dict = {"AdaBoost": 1, "DT": 2, "SVM": 3, "LR": 4, "NB": 5}
                    output_data_path = os.path.join(os.getcwd(),'Noise_'+str(Noise_Ratio*100)+'%'+'MMMultiingle_Traditional'+Normalization_Method)
                    #output_data_path = os.path.join(os.getcwd(), 'Window_Size_' + str(window_size) + '_LS_' + str(lstm_size) + Normalization_Method)

                    if not os.path.isdir(output_data_path):
                        os.makedirs(output_data_path)
                    Main(Bin_or_Multi_Label,Method_Dict,eachfile,window_size_label,lstm_size,Noise_Ratio)

    print(time.time() - start)

