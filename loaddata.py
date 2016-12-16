#_author_by_MC@20160424
import pywt
import matplotlib.pyplot as plt
import time
import random
import shutil
import os
start = time.time()
import numpy as np
from numpy import *
from sklearn.preprocessing import LabelEncoder
from sklearn import svm,preprocessing
from keras.utils import np_utils
import matplotlib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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
    input_data_path = os.path.join(os.getcwd(),'BGP_Data')
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
                if val[-1].strip()== negative_flag:
                    val[-1] = negative_sign
                else:
                    val[-1] = positive_sign
                try:
                    val=list(map(lambda a:float(a),val))
                except:
                    val=list(map(lambda a:str(a),val))

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
    interval = 1
    index = 0
    newdata_count = 0
    initial_value = -999
    while index+window_size < L:
        newdata.append(initial_value)
        newlabel.append(initial_value)
        sequence = []
        for i in range(window_size):
            sequence.append(data[index+i])
            newlabel[newdata_count] = label[index+i]
        index += interval
        newdata[newdata_count]=sequence
        newdata_count += 1
    return np.array(newdata),np.array(newlabel)

def Manipulation(trainX,trainY,time_scale_size):
    window_size = len(trainX[0])
    temp = []
    N = window_size/time_scale_size
    for i in range(len(trainX)):
        temp.append([])
        for j in range(N):
            _value = np.zeros((1,len(trainX[0][0])))
            for k in range(time_scale_size):
                _value += trainX[i][j*time_scale_size+k]
            _value = _value/time_scale_size
            temp[i].extend(list(_value[0]))
    return temp,trainY
def returnPositiveIndex(data,negative_sign):
    temp = []
    for i in range(len(data)):
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
        _index = [i for i in range(len(Y))]
        pos_index = returnPositiveIndex(Y,negative_sign=1)
        pos_index1,pos_index2 = returnSub_Positive(pos_index)

        plt.plot(_index,X[:,0][:,selected_feature],'b-',linewidth=2.0)
        plt.plot(pos_index1,X[pos_index1,0][:,selected_feature],'r-',linewidth=2.0)
        plt.plot(pos_index2,X[pos_index2,0][:,selected_feature],'r-',linewidth=2.0)
        plt.tick_params(labelsize=14)

        if selected_feature == 1:
            plt.ylim(-2,14)
        else:
            plt.ylim(-4,12)

        plt.savefig(os.path.join(os.path.join(os.getcwd(),output_folder),"F_"+str(selected_feature)\
                        +"_AAA"+str(random.randint(1,10000))+"AAAAAA.png"),dpi=100)

def MyEvaluation(Y_Testing, Result):
    acc = 0
    if len(Y_Testing)!=len(Result):
        print("Error!!!")
    else:
        for tab1 in range(len(Result)):
            temp_result = list(map(lambda a:int(round(a)),Result[tab1]))
            temp_true = list(map(lambda a:int(round(a)),Y_Testing[tab1]))
            if list(temp_result) == list(temp_true):
                acc += 1
    return round(float(acc)/len(Result),3)

def Multi_Scale_Wavelet(trainX,trainY,level,is_multi=True,wave_type='db1'):
    temp = [[] for i in range(level)]
    N = trainX.shape[0]
    if (is_multi == True) and (level > 1):
        for i in range(level):
            x = []
            for _feature in range(len(trainX[0])):
                coeffs = pywt.wavedec(trainX[:,_feature], wave_type, level=level)
                current_level = level - i
                for j in range(i+1,level+1):
                    coeffs[j] = None
                _rec = pywt.waverec(coeffs, wave_type)
                x.append(_rec[:N])

            temp[current_level - 1].extend(np.transpose(np.array(x)))

    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(trainX)

    return  np.array(temp),trainX,trainY

def Multi_Scale_Wavelet0(trainX,trainY,level,is_multi=True):
    temp = [[] for i in range(level)]
    x = np.transpose(trainX)
    if is_multi == True:
        for i in range(level):
            coeffs = pywt.wavedec(x[:,0], 'db1', level=level)
            current_level = level - i
            for j in range(i+1,level+1):
                coeffs[j] = None
            _rec = pywt.waverec(coeffs, 'db1')
            temp[current_level-1].extend(np.transpose(_rec))
    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(np.transpose(x))

    return  np.array(temp),trainX,trainY

def Multi_Scale_Plotting(dataMulti,dataA):

    selected_feature = 1
    original = dataA[:,selected_feature]
    original_level1 = dataMulti[0][:,selected_feature]

    level2_A = dataMulti[1][:,selected_feature]
    #Obtain Level_2_D
    coeffs = pywt.wavedec(original, 'db1', level=1)
    newCoeffs = [None,coeffs[1]]
    level2_D = pywt.waverec(newCoeffs,'db1')

    level3_A = dataMulti[2][:,selected_feature]
    #Obtain Level_3_D
    coeffs = pywt.wavedec(original, 'db1', level=2)
    newCoeffs = [None,coeffs[1],None]
    level3_D = pywt.waverec(newCoeffs,'db1')

    level4_A = dataMulti[3][:,selected_feature]
    #Obtain Level_4_D
    coeffs = pywt.wavedec(original, 'db1', level=3)
    newCoeffs = [None,coeffs[1],None,None]
    level4_D = pywt.waverec(newCoeffs,'db1')

    x = [i + 1 for i in range(len(original_level1))]
    #fig = plt.figure(figsize=(20,10),dpi=400)
    fig = plt.figure()
    ymax = 2
    ymin = -2

    #ax1 = fig.add_subplot(4
    # 21)
    ax1 = fig
    plt.plot(x,level4_D,'g')


    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Average Unique AS path length",fontsize=12)
    plt.tick_params(labelsize=12)
    plt.ylim(ymin,ymax)
    plt.grid(True)
    plt.savefig("Level_4_D.png",dpi=800)
    plt.show()

    ax1 = fig.add_subplot(422)
    ax1.plot(x,original_level1,'b')
    plt.xlabel("(b)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(423)
    ax1.plot(x,level2_A,'b')
    plt.xlabel("(c)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(424)
    ax1.plot(x,level2_D,'b')
    plt.xlabel("(d)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)

    ax1 = fig.add_subplot(425)
    ax1.plot(x,level3_A,'b')
    plt.xlabel("(e)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(426)
    ax1.plot(x,level3_D,'b')
    plt.xlabel("(f)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)

    ax1 = fig.add_subplot(427)
    ax1.plot(x,level4_A,'b')
    plt.xlabel("(g)")
    plt.ylabel("Average AS path length")
    plt.ylim(ymin,ymax)
    ax1.grid(True)

    ax1 = fig.add_subplot(428)
    ax1.plot(x,level4_D,'b')
    plt.xlabel("(h)")
    plt.ylabel("Average AS path length")
    ax1.grid(True)
    plt.show()

def Convergge(trainX,trainY,time_scale_size=1):
    window_size = len(trainX[0])
    temp = []
    N = window_size/time_scale_size
    for i in range(len(trainX)):
        temp.append([])
        for j in range(N):
            _value = np.zeros((1,len(trainX[0][0])))
            for k in range(time_scale_size):
                _value += trainX[i][j*time_scale_size+k]
            _value = _value/time_scale_size
            temp[i].append(list(_value[0]))

    return  np.array(temp),trainY
def Fun(multiscaleSequences,case = 'max pooling'):
    temp = []
    if case == 'diag pooling':
        if not len(multiscaleSequences) == len(multiscaleSequences[0]):
            print("-------------------------------------error!")
        else:
            for tab in range(len(multiscaleSequences)):
                temp.append(list(multiscaleSequences[tab][tab]))
    else:
        scale = len(multiscaleSequences)
        sequence_window = len(multiscaleSequences[0])
        dimensions = len(multiscaleSequences[0][0])
        for i in range(sequence_window):
            l = []
            for tab_scale in range(scale):
                l.append(multiscaleSequences[tab_scale][i])
            l = np.array(l)
            temp_sample = []
            for j in range(dimensions):
                if case == 'max pooling':
                    temp_sample.append(np.max(l[:,j]))# max pooling
                elif case == 'mean pooling':
                    temp_sample.append(np.mean(l[:,j]))# max pooling
            temp.append(temp_sample)
    return temp


def Add_Noise(ratio,data):
    w = 5
    x = data[:,:-1]
    _std = x.std(axis=0,ddof=0)
    N = int(ratio*len(data))
    noise = []
    for i in range(N):
        baseinstance_index = random.randint(0,len(data)-1)
        base_instance = data[baseinstance_index]
        noise.append([])
        for j in range(len(_std)):
            temp = random.uniform(_std[j]*-1,_std[j])
            noise[i].append(float(base_instance[j]+temp/w))
        noise[i].append(base_instance[-1])
    noise = np.array(noise)
    return np.concatenate((data,noise),axis=0)
def Mix_Multi_Scale1(trainX_multi,trainY,pooling_type):
    scale = len(trainX_multi)
    length = len(trainX_multi[0])
    temp_trainX = []
    for tab_length in range(length):
        total_ = []
        for tab_scale in range(scale):
            a = trainX_multi[tab_scale][tab_length]
            total_.append(a)
        b = Fun(total_,pooling_type)
        temp_trainX.append(b)
    return np.array(temp_trainX),trainY

def returnTabData(current_cv,cross_cv,dataX,dataY):
    global positive_sign,negative_sign

    positive_index = returnPositiveIndex(dataY, negative_sign)
    negative_index = returnNegativeIndex(dataY, negative_sign)

    pos_data = dataX[positive_index]
    neg_data = dataX[negative_index]

    for tab_cross in range(cross_cv):
        if not tab_cross == current_cv: continue
        pos_train_index = []
        pos_test_index = []
        neg_train_index = []
        neg_test_index = []

        for tab_positive in range(len(positive_index)):
            if int((cross_cv - tab_cross - 1) * len(pos_data) / cross_cv) <= tab_positive < int(
                                    (cross_cv - tab_cross) * len(pos_data) / cross_cv):
                pos_test_index.append(positive_index[tab_positive])
            else:
                pos_train_index.append(positive_index[tab_positive])

        for tab_negative in range(len(negative_index)):
            if int((cross_cv - tab_cross - 1) * len(neg_data) / cross_cv) <= tab_negative < int(\
                                    (cross_cv - tab_cross) * len(neg_data) / cross_cv):
                neg_test_index.append(negative_index[tab_negative])
            else:
                neg_train_index.append(negative_index[tab_negative])


        train_index = np.append(neg_train_index, pos_train_index, axis=0)
        train_index.sort()

        train_index = list(map(lambda a: int(a), train_index))
        train_dataX = dataX[train_index, :]
        train_dataY = dataY[train_index]

        test_index = np.append(neg_test_index, pos_test_index, axis=0)
        test_index.sort()

        test_index = list(map(lambda a: int(a), test_index))
        test_dataX = dataX[test_index, :]
        test_dataY = dataY[test_index]

        #min_number = min(len(train_dataX),len(test_dataX))

        print(str(tab_cross + 1) + "th Cross Validation is running and the training size is ")

        return train_dataX,train_dataY,test_dataX,test_dataY



def GetData(Pooling_Type,Is_Adding_Noise,Noise_Ratio,Method,Fila_Path,FileName,Window_Size,Current_CV,Cross_CV,Bin_or_Multi_Label="Bin",Multi_Scale=True,Wave_Let_Scale=-1,Wave_Type="db1",Time_Scale_Size=1):

    global positive_sign,negative_sign,output_folder
    positive_sign = 0
    negative_sign = 1
    output_folder = "output"

    if not os.path.isdir(os.path.join(os.getcwd(),output_folder)):
        os.makedirs(os.path.join(os.getcwd(),output_folder))


    Data_=LoadData(Fila_Path,FileName)
    scaler = preprocessing.StandardScaler()

   # X_,Y_,X_Validation, Y_Validation = returnTabData(0, 3, Data_[:,:-1],Data_[:,-1])
    #X_Validation, Y_Validation = reConstruction(Window_Size, scaler.fit_transform(X_Validation),Y_Validation)

    #Data_ = np.concatenate((X_,np.reshape(Y_,(len(Y_),1))),axis=1)
    #print("aaaaaaaaaaaaaaa")
    #print(Data_.shape)

    #Plotting_Sequence(Data_[:,0], Data_[:,-1])
    if Is_Adding_Noise == True:
        Data_ = Add_Noise(Noise_Ratio,Data_)



    #if Bin_or_Multi_Label=="Multi":np.random.shuffle(PositiveIndex)
    if Multi_Scale == False:
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
            X_Training, Y_Training, X_Testing, Y_Testing = returnTabData(Current_CV, Cross_CV, Data_Levels_X,Data_Levels_Y)
            print("returnTabData_"+str(Data_Levels_X.shape))
            X_Training_Multi_Level_List[tab_level].extend(X_Training)
            X_Testing_Multi_Level_List[tab_level].extend(X_Testing)

    Y_Training_Encoder = LabelEncoder()
    Y_Training_Encoder.fit(Y_Training)
    Y_Training = Y_Training_Encoder.transform(Y_Training)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_Training = np_utils.to_categorical(Y_Training)

    Y_Testing_Encoder = LabelEncoder()
    Y_Testing_Encoder.fit(Y_Testing)
    Y_Testing = Y_Testing_Encoder.transform(Y_Testing)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_Testing = np_utils.to_categorical(Y_Testing)

    if Multi_Scale == False:
        return X_Training, Y_Training, X_Testing, Y_Testing

    A1 = np.array(X_Training_Multi_Level_List).transpose((1,0,2,3))#batch_size, scale_levels, sequence_window, input_dim
    A2 = np.array(X_Testing_Multi_Level_List).transpose((1,0,2,3)) #batch_size, scale_levels, sequence_window, input_dim
    print("Input shape is"+str(A1.shape))
    #return A1,Y_Training,A2,Y_Testing,X_Validation,Y_Validation
    return A1,Y_Training,A2,Y_Testing



def GetData_WithoutS(Is_Adding_Noise,Noise_Ratio,Fila_Path,FileName,Window_Size,Current_CV,Cross_CV,Bin_or_Multi_Label="Bin",Multi_Scale=False,Wave_Let_Scale=2,Time_Scale_Size=1,Normalize=0):

    global positive_sign,negative_sign
    output_folder = "output"

    if not os.path.isdir(os.path.join(os.getcwd(),output_folder)):
        os.makedirs(os.path.join(os.getcwd(),output_folder))
    positive_sign=0
    negative_sign=1
    Data_=LoadData(Fila_Path,FileName)
    #X_,Y_,X_Validation, Y_Validation = returnTabData(0, 4, Data_[:,:-1],Data_[:,-1])
    #Data_ = np.concatenate((X_,np.reshape(Y_,(len(Y_),1))),axis=1)

    if Is_Adding_Noise == True:
        Data_ = Add_Noise(Noise_Ratio,Data_)
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
        Training_Data_Index = list(map(lambda a:int(a),Training_Data_Index))
        Training_Data = Data_[Training_Data_Index,:]
        Testing_Data_Index=np.append(Negative_Data_Index_Testing,Positive_Data_Index_Testing,axis=0)
        Testing_Data_Index.sort()
        Testing_Data_Index = list(map(lambda a:int(a),Testing_Data_Index))
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



