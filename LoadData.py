import tensorflow as tf

#_author_by_MC@20160424
import os
import pywt
import matplotlib.pyplot as plt
import time
import random
import shutil
start = time.time()
import numpy as np
from numpy import *
from sklearn.preprocessing import LabelEncoder
from sklearn import svm,preprocessing
from keras.utils import np_utils
#import seaborn as sns
import matplotlib
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import RFE
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
    print("Sequence Window is "+str(window_size))
    print(np.array(newdata).shape)
    print(np.array(newlabel).shape)
    return np.array(newdata),np.array(newlabel)
    #print(np.array(newdata).shape)
    #print(np.array(newlabel).shape)
    #return np.concatenate((np.array(newdata),np.array(newlabel)),axis = 1)




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
def returnPositiveIndex(data,negative_sign):
    temp = []
    for i in range(len(data)):
        #print("aaa")
        #print(data)
        try:
            if int(data[i]) != negative_sign:
                temp.append(i)
        except:
            if int(data[i,-1]) != negative_sign:
                temp.append(i)
    return temp

def returnNegativeIndex(data,negative_sign):
    temp = []
    for i in range(len(data)):
        try:
            if int(data[i]) == negative_sign:
                temp.append(i)
        except:
            if int(data[i,-1]) == negative_sign:
                temp.append(i)
    return temp
def returnSub_Positive(positive_list):
    temp1 = []
    temp2 = []
    flag = True
    for tab in range(len(positive_list)-1):
        if positive_list[tab+1] - positive_list[tab] < 2 and flag == True:
            temp1.append(positive_list[tab])
        else:
            flag = False
            temp2.append(positive_list[tab+1])
    return temp1,temp2

def Plotting_Sequence(X,Y):
    global output_folder

    for selected_feature in range(0,1):
        X_index = [i for i in range(len(Y))]
        X_pos_index = returnPositiveIndex(Y,negative_sign=1)
        X_pos_index_sub1,X_pos_index_sub2 = returnSub_Positive(X_pos_index)
        X_neg_index = returnPositiveIndex(Y,negative_sign=1)

        #print(X[:,0].shape)
        #print(X.shape)
        #print(X[:,0][:,2])
        #print(X[:,1][:,2])
        #print(X[:,2][:,2])

        fig = plt.figure(figsize=(12,5))
        if 1>0:
            #print(len(X_index))
            #print(len(X[:,0][:,selected_feature]))
            plt.plot(X_index,X[:,0][:,selected_feature],'b-',linewidth=2.0)
            plt.plot(X_pos_index_sub1,X[X_pos_index_sub1,0][:,selected_feature],'r-',linewidth=2.0)
            plt.plot(X_pos_index_sub2,X[X_pos_index_sub2,0][:,selected_feature],'r-',linewidth=2.0)
            plt.tick_params(labelsize=14)
            if selected_feature == 1:
                plt.ylim(-2,14)
            else:
                plt.ylim(-4,12)
        else:
            plt.plot(X_index,X)
            plt.plot(X_pos_index,X[X_pos_index],'r-')

        plt.savefig(os.path.join(os.path.join(os.getcwd(),output_folder),"F_"+str(selected_feature)\
                        +"_AAA"+str(random.randint(1,10000))+"AAAAAA.png"),dpi=100)

def MyEvaluation(Y_Testing, Result):
    acc = 0
    if len(Y_Testing)!=len(Result):
        print("Error!!!")
    else:
        for tab1 in range(len(Result)):
            temp_result = map(lambda a:int(round(a)),Result[tab1])
            temp_true = map(lambda a:int(round(a)),Y_Testing[tab1])
            #print(type(temp_result))
            #print(temp_result)
            #print(temp_true)
            if list(temp_result) == list(temp_true):
                acc += 1
    return round(float(acc)/len(Result),3)

def Multi_Scale_Wavelet(X_Training,Y_Training,Level,Is_Wave_Let=True,Wave_Type='db1'):
    #print(Wave_Type+" processing+++++++++++++++++++++++++++++++++++++++++++++")
    TEMP_XData = [[] for i in range(Level)]
    #print("X_Training")
    NN = X_Training.shape[0]
    #print(X_Training.shape)
    if (Is_Wave_Let == True) and (Level > 1):
        for tab in range(Level):
            X = []
            for each_feature in range(len(X_Training[0])):
                #print(str(each_feature+1)+" is processing"+" length is "+str(len(X_Training[:,each_feature])))
                coeffs = pywt.wavedec(X_Training[:,each_feature], Wave_Type, level=Level)
                current_level = Level - tab
                for tab2 in range(tab+1,Level+1):
                    coeffs[tab2] = None
                X_WAVELET_REC = pywt.waverec(coeffs, Wave_Type)
                #print("recoverying legnth is "+str(len(X_WAVELET_REC)))
                X.append(X_WAVELET_REC[:NN])
                #print(np.transpose(np.array(X)).shape)

            TEMP_XData[current_level - 1].extend(np.transpose(np.array(X)))

        TEMP_XData = np.array(TEMP_XData)
        #print("11111111111")
        #print(TEMP_XData.shape)
    else:
        for tab in range(Level):
            current_level = Level - tab
            TEMP_XData[current_level - 1].extend(X_Training)

        TEMP_XData = np.array(TEMP_XData)
        #print("22222222222222")
        #print(TEMP_XData.shape)
    return  TEMP_XData,X_Training,Y_Training

def Multi_Scale_Wavelet0(X_Training,Y_Training,Level,Is_Wave_Let=True):
    TEMP_XData = [[] for i in range(Level)]
    X = np.transpose(X_Training)
    print(X.shape)
    if Is_Wave_Let == True:
        for tab in range(Level):
            coeffs = pywt.wavedec(X[:,0], 'db1', level=Level)
            current_level = Level - tab
            for tab2 in range(tab+1,Level+1):
                coeffs[tab2] = None
            #print(len(coeffs))
            X_WAVELET_REC = pywt.waverec(coeffs, 'db1')
            X_1 = np.transpose(X_WAVELET_REC)

            TEMP_XData[current_level-1].extend(X_1)

        TEMP_XData = np.array(TEMP_XData)
    else:
        for tab in range(Level):
            current_level = Level - tab
            X_1 = np.transpose(X)
            TEMP_XData[current_level - 1].extend(X_1)

        TEMP_XData = np.array(TEMP_XData)

    return  TEMP_XData,X_Training,Y_Training

def Multi_Scale_Plotting(Data_Multi,Data_A):

    Selected_Feature = 1
    Original = Data_A[:,Selected_Feature]
    print(Data_Multi[0].shape)
    Original_Or_Level_1 = Data_Multi[0][:,Selected_Feature]

    Level_2_A = Data_Multi[1][:,Selected_Feature]
    #Obtain Level_2_D
    coeffs = pywt.wavedec(Original, 'db1', level=1)
    NewCoeffs = [None,coeffs[1]]
    Level_2_D = pywt.waverec(NewCoeffs,'db1')
    #Level_2_D = R1[:,Selected_Feature]

    Level_3_A = Data_Multi[2][:,Selected_Feature]
    #Obtain Level_3_D
    coeffs = pywt.wavedec(Original, 'db1', level=2)
    NewCoeffs = [None,coeffs[1],None]
    Level_3_D = pywt.waverec(NewCoeffs,'db1')
    #Level_3_D = R2[:,Selected_Feature]

    Level_4_A = Data_Multi[3][:,Selected_Feature]

    #Obtain Level_4_D
    coeffs = pywt.wavedec(Original, 'db1', level=3)
    NewCoeffs = [None,coeffs[1],None,None]
    Level_4_D = pywt.waverec(NewCoeffs,'db1')
    #Level_4_D = R3[:,Selected_Feature]

    X = [i + 1 for i in range(len(Original_Or_Level_1))]
    #fig = plt.figure(figsize=(20,10),dpi=400)
    fig = plt.figure()
    ymax = 2
    ymin = -2

    #ax1 = fig.add_subplot(4
    # 21)
    ax1 = fig
    plt.plot(X,Level_4_D,'g')


    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Average Unique AS path length",fontsize=12)
    plt.tick_params(labelsize=12)
    plt.ylim(ymin,ymax)
    plt.grid(True)
    plt.savefig("Level_4_D.png",dpi=800)
    plt.show()

    ax1 = fig.add_subplot(422)
    ax1.plot(X,Original_Or_Level_1,'b')
    plt.xlabel("(b)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(423)
    ax1.plot(X,Level_2_A,'b')
    plt.xlabel("(c)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(424)
    ax1.plot(X,Level_2_D,'b')
    plt.xlabel("(d)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)

    ax1 = fig.add_subplot(425)
    ax1.plot(X,Level_3_A,'b')
    plt.xlabel("(e)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(426)
    ax1.plot(X,Level_3_D,'b')
    plt.xlabel("(f)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)

    ax1 = fig.add_subplot(427)
    ax1.plot(X,Level_4_A,'b')
    plt.xlabel("(g)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(428)
    ax1.plot(X,Level_4_D,'b')
    plt.xlabel("(h)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)

    plt.show()

    #plt.savefig("C.pdf",dpi=400)

    #plt.show()


def Convergge(X_Training,Y_Training,time_scale_size=1):
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

    #print("Converge")
    #print(np.array(TEMP_XData).shape)

    return  np.array(TEMP_XData),Y_Training
def Fun(Multiscale_Sequence_List,case = 'max pooling'):
    Temp = []
    if case == 'diag pooling':
        if not len(Multiscale_Sequence_List) == len(Multiscale_Sequence_List[0]):
            print("-------------------------------------ERROR!!!")
        else:
            for tab in range(len(Multiscale_Sequence_List)):
                Temp.append(list(Multiscale_Sequence_List[tab][tab]))
    else:
        scale = len(Multiscale_Sequence_List)
        sequence_window = len(Multiscale_Sequence_List[0])
        dimensions = len(Multiscale_Sequence_List[0][0])
        for tab in range(sequence_window):
            AAA = []
            for tab_scale in range(scale):
                AAA.append(Multiscale_Sequence_List[tab_scale][tab])
            AAA = np.array(AAA)
            temp_sample = []
            for tab_dimension in range(dimensions):
                if case == 'max pooling':
                    temp_sample.append(np.max(AAA[:,tab_dimension]))# max pooling
                elif case == 'mean pooling':
                    temp_sample.append(np.mean(AAA[:,tab_dimension]))# max pooling
            Temp.append(temp_sample)
    #print("LALALALLALALA"+str(np.array(Temp).shape))
    return Temp

    """
    B = np.ones(shape=(1,len(Sequence_List)))
    if case == 'max pooling':
        C = np.divide(np.matmul(B,A),len(Sequence_List))[0]
    elif case == 'mean pooling':
        C = A[-1]
    elif case == "diag pooling":
        C = A[tab_scale]
    elif case ==
    return C
    """

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
def Mix_Multi_Scale1(X_Train_Multi_Scale,Y_Train,Pooling_Type):
    #print("Multi Scale is ")
    #print(X_Train_Multi_Scale)
    Scale = len(X_Train_Multi_Scale)
    Length = len(X_Train_Multi_Scale[0])
    Temp_X_Train = []
    for tab_length in range(Length):
        Total_AAA = []
        for tab_scale in range(Scale):
            AAA = X_Train_Multi_Scale[tab_scale][tab_length]
            Total_AAA.append(AAA)
        BBB = Fun(Total_AAA,Pooling_Type)
        Temp_X_Train.append(BBB)
    return np.array(Temp_X_Train),Y_Train

def returnTabData(Current_CV,Cross_CV,Data_X,Data_Y):
    global positive_sign,negative_sign

    positive_index = returnPositiveIndex(Data_Y, negative_sign)
    negative_index = returnNegativeIndex(Data_Y, negative_sign)

    pos_data = Data_X[positive_index]
    neg_data = Data_X[negative_index]

    for tab_cross in range(Cross_CV):
        if not tab_cross == Current_CV: continue
        positive_Data_index_Train = []
        positive_Data_index_Test = []
        negative_Data_index_Train = []
        negative_Data_index_Test = []

        for tab_positive in range(len(positive_index)):
            if int((Cross_CV - tab_cross - 1) * len(pos_data) / Cross_CV) <= tab_positive < int(
                                    (Cross_CV - tab_cross) * len(pos_data) / Cross_CV):
                positive_Data_index_Test.append(positive_index[tab_positive])
            else:
                positive_Data_index_Train.append(positive_index[tab_positive])

        for tab_negative in range(len(negative_index)):
            if int((Cross_CV - tab_cross - 1) * len(neg_data) / Cross_CV) <= tab_negative < int(\
                                    (Cross_CV - tab_cross) * len(neg_data) / Cross_CV):
                negative_Data_index_Test.append(negative_index[tab_negative])
            else:
                negative_Data_index_Train.append(negative_index[tab_negative])


        Training_Data_Index = np.append(negative_Data_index_Train, positive_Data_index_Train, axis=0)
        Training_Data_Index.sort()
        #print(Training_Data_Index)
        #print(len(negative_Data_index_Train))
        #print(len(positive_Data_index_Train))
        #print(len(negative_index))
        #print(len(positive_index))
        Training_Data_Index = map(lambda a: int(a), Training_Data_Index)
        Training_Data_X = Data_X[Training_Data_Index, :]
        Training_Data_Y = Data_Y[Training_Data_Index]

        Testing_Data_Index = np.append(negative_Data_index_Test, positive_Data_index_Test, axis=0)
        Testing_Data_Index.sort()
        #print(Testing_Data_Index)
        #print(len(negative_Data_index_Test))
        #print(len(positive_Data_index_Test))

        Testing_Data_Index = map(lambda a: int(a), Testing_Data_Index)
        Testing_Data_X = Data_X[Testing_Data_Index, :]
        Testing_Data_Y = Data_Y[Testing_Data_Index]

        min_number = min(len(Training_Data_X),len(Testing_Data_X))

        print(str(tab_cross + 1) + "th Cross Validation is running and the training size is " + \
            str(len(Training_Data_X)) + ", testing size is " + str(len(Testing_Data_Y)) + "......")

        #Plotting_Sequence(Training_Data_X[0:min_number],Training_Data_Y[0:min_number])
        return Training_Data_X[0:min_number],Training_Data_Y[0:min_number],Testing_Data_X[0:min_number],Testing_Data_Y[0:min_number]




def GetData(Pooling_Type,Is_Adding_Noise,Noise_Ratio,Method,Fila_Path,FileName,Window_Size,Current_CV,Cross_CV,Bin_or_Multi_Label="Bin",Multi_Scale=True,Wave_Let_Scale=-1,Wave_Type="db1",Time_Scale_Size=1):

    global positive_sign,negative_sign,output_folder
    positive_sign = 0
    negative_sign = 1
    output_folder = "ABC"

    if not os.path.isdir(os.path.join(os.getcwd(),output_folder)):
        os.makedirs(os.path.join(os.getcwd(),output_folder))
    else:
        shutil.rmtree(os.path.join(os.getcwd(),output_folder))
        os.makedirs(os.path.join(os.getcwd(),output_folder))


    Data_=LoadData(Fila_Path,FileName)
    #Plotting_Sequence(Data_[:,0], Data_[:,-1])

    if Is_Adding_Noise == True:
        Data_ = Add_Noise(Noise_Ratio,Data_)


    #Data_ = Add_Noise(Noise_Ratio,Data_)
    #Pos_Data=Data_[Data_[:,-1]!=negative_sign]
    #Neg_Data=Data_[Data_[:,-1]==negative_sign]
    scaler = preprocessing.StandardScaler()

    #if Bin_or_Multi_Label=="Multi":np.random.shuffle(PositiveIndex)
    if Wave_Let_Scale < 0:
        Data_Sequenlized_X,Data_Sequenlized_Y = reConstruction(Window_Size, scaler.fit_transform(Data_[:, :-1]), Data_[:, -1])
        X_Training, Y_Training,X_Testing,Y_Testing = returnTabData(Current_CV,Cross_CV,Data_Sequenlized_X,Data_Sequenlized_Y)

    else:

        Scale_Level = Wave_Let_Scale
        X_Training_Multi_Level_List = [[] for i in range(Scale_Level)]
        X_Testing_Multi_Level_List = [[] for i in range(Scale_Level)]

        print("wave type is "+ str(Wave_Type)+" and the scale is "+str(Scale_Level))

        Data_Multi_Level_X, Data_X, Data_Y = Multi_Scale_Wavelet(Data_[:, :-1],Data_[:, -1],Scale_Level,True,Wave_Type)

        #Plot the different scale
        #Multi_Scale_Plotting(Data_Multi_Level_X,Data_X)


        for tab_level in range(Scale_Level):
            Data_Levels_X,Data_Levels_Y = reConstruction(Window_Size, scaler.fit_transform(Data_Multi_Level_X[tab_level]), Data_Y)

            #(X_Testing,Y_Testing) = reConstruction(Window_Size, scaler.fit_transform(X_Testing_Multi[tab_level]), Y_Testing_1)
            #(X_Testing,Y_Testing) = reConstruction(Window_Size, (X_Testing_Multi[tab_level]), Y_Testing_1)
            X_Training, Y_Training, X_Testing, Y_Testing = returnTabData(Current_CV, Cross_CV, Data_Levels_X,Data_Levels_Y)
            print("returnTabData_"+str(Data_Levels_X.shape))
            X_Training_Multi_Level_List[tab_level].extend(X_Training)
            X_Testing_Multi_Level_List[tab_level].extend(X_Testing)


        #(X_Training_Single_Sclae, Y_Training) = reConstruction(Window_Size, scaler.fit_transform(X_Training_1), Y_Training_1)
        #(X_Training_Single_Sclae, Y_Training) = reConstruction(Window_Size, X_Training_1, Y_Training_1)
        #(X_Testing_Single_Sclae, Y_Testing) = reConstruction(Window_Size, scaler.fit_transform(X_Testing_1), Y_Testing_1)
        #(X_Testing_Single_Sclae, Y_Testing) = reConstruction(Window_Size, X_Testing_1, Y_Testing_1)

    Y_Training0 = Y_Training
    Y_Training_Encoder = LabelEncoder()
    Y_Training_Encoder.fit(Y_Training)
    encoded_Y1 = Y_Training_Encoder.transform(Y_Training)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_Training = np_utils.to_categorical(encoded_Y1)

    Y_Testing0 = Y_Testing
    Y_Testing_Encoder = LabelEncoder()
    Y_Testing_Encoder.fit(Y_Testing)
    encoded_Y2 = Y_Testing_Encoder.transform(Y_Testing)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_Testing = np_utils.to_categorical(encoded_Y2)

    #print("hahaha")
    #print(X_Testing_Single_Sclae)
    if Multi_Scale == False:
        #print("This is single ....................")
        if Wave_Let_Scale < 0:
            print("XXXXXX_Original" + str(X_Training.shape))
            return X_Training, Y_Training, X_Testing, Y_Testing
        else:
            print("XXXXXX_Scale" + str(Wave_Let_Scale))
            print(np.array(X_Training_Multi_Level_List[-1]).shape)
            print(np.array(X_Training_Multi_Level_List[-1])[0])

            return np.array(X_Training_Multi_Level_List[-1]), Y_Training, X_Testing, Y_Testing
    #if Wave_Let_Scale > 0 and Method == "Attention222":
        #X_Training_Multi_Level_List1,Y_Training = Mix_Multi_Scale1(X_Training_Multi_Level_List,Y_Training,Pooling_Type)
        #X_Testing_Multi_Level_List1, Y_Training = Mix_Multi_Scale1(X_Testing_Multi_Level_List, Y_Training,Pooling_Type)
    #try:
        #print("***********************"+str(len(X_Testing_Multi_Level_List2)))
    #except:
        #pass

    print("XXXXXXXXXXXXXXXXXXX"+str(np.array(X_Training_Multi_Level_List).shape))
    print("XXXXXXXXXXXXXXXXXXX"+str(np.array(X_Testing_Multi_Level_List).shape))

    return np.array(X_Training_Multi_Level_List),Y_Training,np.array(X_Testing_Multi_Level_List),Y_Testing



def GetData_WithoutS(Is_Adding_Noise,Noise_Ratio,Fila_Path,FileName,Window_Size,Current_CV,Cross_CV,Bin_or_Multi_Label="Bin",Multi_Scale=False,Wave_Let_Scale=2,Time_Scale_Size=1,Normalize=0):

    global positive_sign,negative_sign
    positive_sign=0
    negative_sign=1
    Data_=LoadData(Fila_Path,FileName)
    if Is_Adding_Noise == True:
        Data_ = Add_Noise(Noise_Ratio,Data_)
    #Pos_Data=Data_[Data_[:,-1]!=negative_sign]
    #Neg_Data=Data_[Data_[:,-1]==negative_sign]
    if Normalize == 1 or  Normalize==10 or Normalize==11:
        scaler = preprocessing.MinMaxScaler()
    elif Normalize == 2:
        scaler = preprocessing.Normalizer()
    else:
        scaler = preprocessing.StandardScaler()
    PositiveIndex = returnPositiveIndex(Data_,negative_sign)
    NegativeIndex = returnNegativeIndex(Data_,negative_sign)

    if Bin_or_Multi_Label=="Multi":np.random.shuffle(PositiveIndex)
    Pos_Data = Data_[PositiveIndex]
    Neg_Data = Data_[NegativeIndex]

    for tab_cross in range(Cross_CV):
        if not tab_cross == Current_CV: continue

        Positive_Data_Index_Training=[]
        Positive_Data_Index_Testing=[]
        Negative_Data_Index_Training=[]
        Negative_Data_Index_Testing=[]

        for tab_positive in range(len(PositiveIndex)):
            if int((Cross_CV-tab_cross-1)*len(Pos_Data)/Cross_CV)<=tab_positive<int((Cross_CV-tab_cross)*len(Pos_Data)/Cross_CV):
                Positive_Data_Index_Testing.append(PositiveIndex[tab_positive])
            else:
                Positive_Data_Index_Training.append(PositiveIndex[tab_positive])

        for tab_negative in range(len(NegativeIndex)):
            if int((Cross_CV-tab_cross-1)*len(Neg_Data)/Cross_CV)<=tab_negative<int((Cross_CV-tab_cross)*len(Neg_Data)/Cross_CV):
                Negative_Data_Index_Testing.append(NegativeIndex[tab_negative])
            else:
                Negative_Data_Index_Training.append(NegativeIndex[tab_negative])


        Training_Data_Index=np.append(Negative_Data_Index_Training,Positive_Data_Index_Training,axis=0)
        Training_Data_Index.sort()
        Training_Data_Index = map(lambda a:int(a),Training_Data_Index)
        Training_Data = Data_[Training_Data_Index,:]
        Testing_Data_Index=np.append(Negative_Data_Index_Testing,Positive_Data_Index_Testing,axis=0)
        Testing_Data_Index.sort()
        Testing_Data_Index = map(lambda a:int(a),Testing_Data_Index)
        Testing_Data = Data_[Testing_Data_Index,:]


        print(str(tab_cross + 1) + "th Cross Validation>>>>>>>> is running and the training size is " + str(
            len(Training_Data)) + ", testing size is " + str(len(Testing_Data)) + "......"+str(Normalize))


        #X_Training = np.array(scaler.fit_transform(Training_Data[:, :-1]))
        #Y_Training = np.array(Training_Data[:, -1])

        #X_Testing = np.array(scaler.fit_transform(Testing_Data[:, :-1]))
        #Y_Testing = np.array(Testing_Data[:, -1])

        if Normalize == 5:#SVMF
            if 'AS' in FileName:
                num_selected_features = 27# AS leak
            num_selected_features = 3

            X_Training = SelectKBest(f_classif, k=num_selected_features).fit_transform(Training_Data[:, :-1], Training_Data[:, -1])
            X_Testing = SelectKBest(f_classif, k=num_selected_features).fit_transform(Testing_Data[:, :-1], Testing_Data[:, -1])

        elif Normalize == 10:#NBF
            num_selected_features = 12
            X_Training = SelectKBest(f_classif, k=num_selected_features).fit_transform(Training_Data[:, :-1], Training_Data[:, -1])
            X_Testing = SelectKBest(f_classif, k=num_selected_features).fit_transform(Testing_Data[:, :-1], Testing_Data[:, -1])

        else:
            X_Training = Training_Data[:, :-1]
            X_Testing = Testing_Data[:, :-1]

        X_Training = np.array(scaler.fit_transform(X_Training))
        Y_Training = np.array(Training_Data[:, -1])

        X_Testing = np.array(scaler.fit_transform(X_Testing))
        Y_Testing = np.array(Testing_Data[:, -1])



        Y_Training_Encoder = LabelEncoder()
        Y_Training_Encoder.fit(Y_Training)
        encoded_Y1 = Y_Training_Encoder.transform(Y_Training)
        # convert integers to dummy variables (i.e. one hot encoded)
        Y_Training = np_utils.to_categorical(encoded_Y1)

        Y_Testing_Encoder = LabelEncoder()
        Y_Testing_Encoder.fit(Y_Testing)
        encoded_Y2 = Y_Testing_Encoder.transform(Y_Testing)
        # convert integers to dummy variables (i.e. one hot encoded)
        Y_Testing = np_utils.to_categorical(encoded_Y2)
        return X_Training,Y_Training,np.array(Training_Data[:, -1]),X_Testing,Y_Testing,np.array(Testing_Data[:, -1])


def get_all_subfactors(number):
    temp_list = []
    temp_list.append(1)
    temp_list.append(2)
    for i in range(3,number):
        if number%i == 0 :
            temp_list.append(i)
    temp_list.append(number)
    return temp_list


def set_style():
    plt.style.use(['seaborn-dark', 'seaborn-paper'])
    matplotlib.rc("font", family="serif")
if __name__=='__main__':
    global filepath, filename, fixed_seed_num, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv
    # ---------------------------Fixed Parameters------------------------------
    filepath = os.getcwd()
    #set_style()
    sequence_window = 10
    hidden_units = 200
    input_dim = 33
    number_class = 2
    cross_cv = 2
    fixed_seed_num = 1337
    # -------------------------------------------------------------------------
    filename = 'HB_AS_Leak.txt'
    learning_rate = 0.01
    epoch = 50
    case_list = [1]
    multi_scale_value = sequence_window
    #os.chdir("/home/grads/mcheng223/IGBB")
    positive_sign=0
    negative_sign=1
    ACCURACY=[]
    sequence_window_list = [10,20,30]
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

    learning_rate = 0.001

    epoch = 200
    case_list = [0,1,2]
    tab_cv = 0
    wave_type = 'db1'
    pooling_type = 'max pooling'

    #x_train_multi_list,x_train, y_train, x_testing_multi_list,x_test, y_test = GetData(pooling_type,'Attention',filepath, filename, sequence_window,tab_cv,cross_cv,Multi_Scale=is_multi_scale,Wave_Let_Scale=training_level)


    for eachfile in filename_list:
        if '.py' in eachfile or '.DS_' in eachfile: continue
        if '.txt' in eachfile:
            pass
        else:
            continue
        if  not eachfile=='HB_AS_Leak.txt':continue
        else:
            training_level = 10
            is_multi_scale = True
        print(eachfile+ " is processing---------------------------------------------------------------------------------------------")
        x_train, y_train,x_test, y_test = GetData(pooling_type,is_add_noise,noise_ratio,'Attention',filepath, filename, sequence_window,tab_cv,cross_cv,Multi_Scale=is_multi_scale,Wave_Let_Scale=training_level,Wave_Type=wave_type)

        #for lstm_size in lstm_size_list:
            #for window_size_label in window_size_label_list:
                #if window_size_label == 'true':
                    #Method_Dict={"LSTM": 0}
                    #Method_Dict = {"LSTM": 0,"AdaBoost": 1, "DT": 2, "SVM": 3, "LOGREG": 4, "NB": 5}
                    #Method_Dict = {"MSLSTM":0,"LSTM": 0,"AdaBoost": 1, "SVM": 3, "NB": 5}

                    #for window_size in window_size_list:
                        #time_scale_size_list = get_all_subfactors(window_size)
                        #output_data_path = os.path.join(os.getcwd(),'NNNNNBBBBB_'+str(Noise_Ratio*100)+'%'+'SSSingle_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        #output_data_path = os.path.join(os.getcwd(),'NNNNNBBBBB_'+'MMMulti_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        #output_data_path = os.path.join(os.getcwd(),'Noise_'+str(Noise_Ratio*100)+'%'+'SSSingle_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        #output_data_path = os.path.join(os.getcwd(),'Half_Minutes_Window_Size_' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)

                        #if not os.path.isdir(output_data_path):
                            #os.makedirs(output_data_path)
                        #time_scale_size_list = [2,3]
                        #for time_scale_size in time_scale_size_list:

                            #if  time_scale_size > 1: continue
                            #GetData(Bin_or_Multi_Label,Method_Dict,eachfile,window_size_label,lstm_size,Noise_Ratio,window_size,time_scale_size)

                #else:
                    #continue
                    #Method_Dict={"NB": 5}

                    #Method_Dict = {"AdaBoost": 1, "DT": 2, "SVM": 3, "LR": 4, "NB": 5}
                    #output_data_path = os.path.join(os.getcwd(),'Noise_'+str(Noise_Ratio*100)+'%'+'MMMultiingle_Traditional'+Normalization_Method)
                    #output_data_path = os.path.join(os.getcwd(), 'Window_Size_' + str(window_size) + '_LS_' + str(lstm_size) + Normalization_Method)

                    #if not os.path.isdir(output_data_path):
                        #os.makedirs(output_data_path)
                    #GetData(Bin_or_Multi_Label,Method_Dict,eachfile,window_size_label,lstm_size,Noise_Ratio)

    #print(time.time() - start)



