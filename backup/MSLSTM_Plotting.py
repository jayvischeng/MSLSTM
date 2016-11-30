#_author_by_MC@20160424
import os
import time
start = time.time()
import pywt
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
global A0,A1,A2,A3,A4,D1,D2,D3,Original
Original=[]
A0=[]
A1=[]
A2=[]
A3=[]
A4=[]
D1=[]
D2=[]
D3=[]
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
            #print(type(temp_result))
            #print(temp_result)
            #print(temp_true)
            if list(temp_result) == list(temp_true):
                acc += 1
    return round(float(acc)/len(Result),3)
def ReturnNewCoeff(coeffs,level):
    N =len(coeffs)
    #NewCoeffs = [coeffs[0]]
    NewCoeffs = [None for i in range(N)]
    if level < N:
        for tab in range(0,N):
            if tab <= level:
                NewCoeffs[tab] = coeffs[tab]
            else:
                NewCoeffs[tab] = None

    else:
        NewCoeffs = [coeffs[0]]
        for tab in range(1,N):
            NewCoeffs.append(coeffs[tab])



    return NewCoeffs
def Plottint(Data_A,Count,Total):
    m,n = Data_A.shape
    if min(m,n) < 8:
        number = 2
    elif min(m,n)< 16:
        number = 3
    elif min(m,n)<= 32:
        number = 4
    print(Data_A.shape)
    coeffs = pywt.wavedec(Data_A, 'db0', level=5)
    #X = [i + 1 for i in range(len(Data_A[0))]

    Selected_Feature = 4

    #print(len(coeffs))

    if number == 4 and (Total-Count)> 32:
        Original.extend(Data_A[Selected_Feature])
        NewCoeffs = ReturnNewCoeff(coeffs,0)
        R0 = pywt.waverec(NewCoeffs,'db1')
        A4.extend(R0[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,1)
        R1 = pywt.waverec(NewCoeffs,'db1')
        A3.extend(R1[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,2)
        R2 = pywt.waverec(NewCoeffs,'db1')
        A2.extend(R2[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,3)
        R3 = pywt.waverec(NewCoeffs,'db1')
        A1.extend(R3[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,4)
        R4 = pywt.waverec(NewCoeffs,'db1')
        A0.extend(R4[Selected_Feature])

    else:
        X = [i + 1 for i in range(len(Original))]
        plt.plot(X,Original,'b')
        max_ = np.max(Original) + 0.1
        plt.xlabel("Time")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        plt.grid(True)
        plt.savefig("D4.pdf",dpi=400)
        plt.show()



        fig = plt.figure(figsize=(20,10),dpi=400)

        ax1 = fig.add_subplot(321)
        ax1.plot(X,Original,'b')
        plt.xlabel("(a)")
        plt.ylabel("Average AS path length")
        max_ = np.max(Original) + 0.1
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(322)
        ax1.plot(X,A4,'b')
        plt.xlabel("(b)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(323)
        ax1.plot(X,A3,'b')
        plt.xlabel("(c)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(324)
        ax1.plot(X,A2,'b')
        plt.xlabel("(d)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(325)
        ax1.plot(X,A1,'b')
        plt.xlabel("(e)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(326)
        ax1.plot(X,A0,'b')
        plt.xlabel("(f)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)
        plt.savefig("C.png",dpi=400)

        plt.show()

        #plt.savefig("C.pdf",dpi=400)

        #plt.show()

        #print(R1+R2)

def Plottint222(Data_A,Count,Total):
    m,n = Data_A.shape
    if min(m,n) < 8:
        number = 2
    elif min(m,n)< 16:
        number = 3
    elif min(m,n)<= 32:
        number = 4
    coeffs1 = pywt.wavedec(Data_A, 'db1', level=5)
    #X = [i + 1 for i in range(len(Data_A[0))]

    Selected_Feature = 4

    print(len(coeffs1))

    if number == 4 and (Total-Count)> 32:

        Original.extend(Data_A[Selected_Feature])

        #Obtain D1
        coeffs1 = pywt.wavedec(Data_A, 'db1', level=1)
        NewCoeffs1 = [coeffs1[0],None]
        R1 = pywt.waverec(NewCoeffs1,'db1')
        A1.extend(R1[Selected_Feature])

        #Obtain D2
        coeffs2 = pywt.wavedec(Data_A, 'db1', level=2)
        NewCoeffs2 = [coeffs2[0],None,None]
        R2 = pywt.waverec(NewCoeffs2,'db1')
        A2.extend(R2[Selected_Feature])

        #Obtain D3
        coeffs3 = pywt.wavedec(Data_A, 'db1', level=3)
        NewCoeffs3 = [coeffs3[0],None,None,None]
        R3 = pywt.waverec(NewCoeffs3,'db1')
        A3.extend(R3[Selected_Feature])

        #Obtain D4
        coeffs4 = pywt.wavedec(Data_A, 'db1', level=4)
        NewCoeffs4 = [coeffs4[0],None,None,None,None]
        R4 = pywt.waverec(NewCoeffs4,'db1')
        A4.extend(R4[Selected_Feature])
        """
        NewCoeffs = ReturnNewCoeff(coeffs,0)
        R0 = pywt.waverec(NewCoeffs,'db1')
        A4.extend(R0[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,1)
        R1 = pywt.waverec(NewCoeffs,'db1')
        A3.extend(R1[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,2)
        R2 = pywt.waverec(NewCoeffs,'db1')
        A2.extend(R2[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,3)
        R3 = pywt.waverec(NewCoeffs,'db1')
        A1.extend(R3[Selected_Feature])

        NewCoeffs = ReturnNewCoeff(coeffs,4)
        R4 = pywt.waverec(NewCoeffs,'db1')
        A0.extend(R4[Selected_Feature])
        """
    else:

        X = [i + 1 for i in range(len(Original))]
        #print(len(X))
        #print(len(A1))
        print(A3)
        plt.plot(X,Original,'b')
        max_ = np.max(Original) + 0.1
        #max_ = 0.5


        plt.xlabel("Time")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        plt.grid(True)
        plt.savefig("D4.pdf",dpi=400)
        plt.show()


        """
        fig = plt.figure(figsize=(20,10),dpi=400)

        ax1 = fig.add_subplot(321)
        ax1.plot(X,Original,'b')
        plt.xlabel("(a)")
        plt.ylabel("Average AS path length")
        max_ = np.max(Original) + 0.1
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(322)
        ax1.plot(X,A4,'b')
        plt.xlabel("(b)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(323)
        ax1.plot(X,A3,'b')
        plt.xlabel("(c)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(324)
        ax1.plot(X,A2,'b')
        plt.xlabel("(d)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(325)
        ax1.plot(X,A1,'b')
        plt.xlabel("(e)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)

        ax1 = fig.add_subplot(326)
        ax1.plot(X,A0,'b')
        plt.xlabel("(f)")
        plt.ylabel("Average AS path length")
        plt.ylim(-1*max_+0.5,max_)
        ax1.grid(True)
        plt.savefig("C.pdf",dpi=400)

        plt.show()
        """
        #plt.savefig("C.pdf",dpi=400)

        #plt.show()

        #print(R1+R2)
def Convergge_BBB(X_Training,Y_Training,time_scale_size,components):
    global COEFF_LIST
    COEFF_LIST = []
    window_size = len(X_Training[0])
    #pca = RandomizedPCA(n_components=components)
    #lda = LinearDiscriminantAnalysis(n_components=2)
    #svd = TruncatedSVD(n_components=components)
    #X_r2 = lda.fit(X, y).transform(X)
    pca = PCA(n_components=components)
    TEMP_XData = []

    for tab1 in range(len(X_Training)):
        TEMP_XData.append([])
        X = np.transpose(X_Training[tab1])
        if tab1%window_size == 0:
            print("----------------------------------------------------------"+str(tab1)+"--------"+str(len(X_Training[tab1])))

            Plottint(X,tab1,len(X_Training)-1)
        else:
            continue
        #coeffs = pywt.dwt(X, 'db3')
        #continue
        coeffs = pywt.wavedec(X, 'db1', level=1)

        X_CA = coeffs[0]
        COEFF_LIST.append(X_CA)

 
    COEFF_LIST = np.array(COEFF_LIST)
    print("COEFF SHAPE IS "+str(X_CA.shape))
    return COEFF_LIST,Y_Training
        #X_CD = coeffs[1]

        #X = np.concatenate((X1,X2),axis=1)
        #y = [Y_Training[tab1] for i in range(len(X_Training[tab1]))]
        #X = lda.fit(X,y).transform(X)
        #X = np.transpose(X)
        #print(X.shape)
        #print(ddd)
    """
        pca.fit(X_CA)
        #print("-------------------------X_CA----------------"+str(X_CA.shape))
        X_WAVELET_PCA = pca.transform(X_CA)
        X_CA_PCA_REC = pca.inverse_transform(X_WAVELET_PCA)
        #print(X_CA_PCA_REC)
        coeffs[0] = X_CA_PCA_REC
        X_WAVELET_REC = pywt.waverec(coeffs, 'db1')
        #X_WAVELET_REC = pywt.idwt(X_CA_PCA_REC,X_CD, 'db1')

        X_1 = np.transpose(X_WAVELET_REC)
        #print(X_1.shape)
        #X = svd.fit_transform(X)
        TEMP_XData[tab1].extend(X_1)
    #print("111111111111111111111111111111111111111111111111111111111111111")
    #print(np.array(TEMP_XData)[0])
    #print("999999999999999999999999999999999999999999999999999999999999999")
    #print(fs)
    return  np.array(TEMP_XData),Y_Training
    """


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
    global COEFF_LIST
    if Normalization_Method == "_Std":
        scaler = preprocessing.StandardScaler()
    elif Normalization_Method == "_L2norm":
        scaler = preprocessing.Normalizer()
    elif Normalization_Method == "_Minmax":
        scaler = preprocessing.MinMaxScaler()

    Data_=LoadData(input_data_path,filename)
    cross_folder = 2
    Data_ = Add_Noise(Noise_Ratio,Data_)
    #Pos_Data=Data_[Data_[:,-1]!=negative_sign]
    #Neg_Data=Data_[Data_[:,-1]==negative_sign]
    PositiveIndex = returnPositiveIndex(Data_,negative_sign)
    NegativeIndex = returnNegativeIndex(Data_,negative_sign)

    if Bin_or_Multi_Label=="Multi":np.random.shuffle(PositiveIndex)
    Pos_Data = Data_[PositiveIndex]
    Neg_Data = Data_[NegativeIndex]

    for tab_cross in range(cross_folder):
        if not tab_cross==0:continue
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
        Training_Data_Index = map(lambda a:int(a),Training_Data_Index)
        Training_Data = Data_[Training_Data_Index,:]
        Testing_Data_Index=np.append(Negative_Data_Index_Testing,Positive_Data_Index_Testing,axis=0)
        Testing_Data_Index.sort()
        Testing_Data_Index = map(lambda a:int(a),Testing_Data_Index)
        Testing_Data = Data_[Testing_Data_Index,:]
        #print(str(tab_cross+1)+"th Cross Validation is running and the training size is "+str(len(Training_Data))+", testing size is "+str(len(Testing_Data))+"......")

        #positive_=Training_Data[Training_Data[:,-1]==positive_sign]
        #negative_=Training_Data[Training_Data[:,-1]==negative_sign]
        #print("IR is :"+str(float(len(negative_))/len(positive_)))
        import pywt
        if window_size_label == "true":
            np.random.seed(1337)  # for reproducibility
            batch_size = 300
            X_Testing_0 = scaler.fit_transform(Testing_Data[:, :-1])
            Y_Testing_0 = Testing_Data[:,-1]
            (X_Training_1, Y_Training_1) = reConstruction(window_size, scaler.fit_transform(Training_Data[:,:-1]),Training_Data[:,-1])
            selected_features = 4



            X_Training, Y_Training = Convergge_BBB(X_Training_1, Y_Training_1, time_scale_size,time_scale_size)
            #X_Training, Y_Training = Convergge(X_Training_1, Y_Training_1, time_scale_size)
            (X_Testing_1, Y_Testing_1) = reConstruction(window_size, scaler.fit_transform(Testing_Data[:,:-1]),Testing_Data[:,-1])

            P0 = X_Testing_0[Y_Testing_0 == 0, :][:, selected_features]
            P0_Index = returnPositiveIndex(Training_Data, 1)
            N0_Index = returnNegativeIndex(Training_Data, 1)
            Y0 = X_Testing_0[:, selected_features]
            X0 = [i for i in range(len(Y0))]
            if time_scale_size == 0:

                plt.plot(X0, Y0, 'b')
                plt.plot(P0_Index, P0, 'r')
                plt.xlim(window_size, len(X0))
                plt.ylim(-0.1, 0.4)
                print("0 is completed!")

            else:
                #X_Testing, Y_Testing = Convergge_BBB(X_Testing_1, Y_Testing_1, time_scale_size,time_scale_size)
                #X_Testing, Y_Testing = Convergge(X_Testing_1, Y_Testing_1, time_scale_size)

                #AAA = MeanPooling(X_Testing[Y_Testing == 0, :])[:, selected_features]
                #BBB = MeanPooling(X_Testing[Y_Testing == 1, :])[:, selected_features]

                #AAA = X_Testing[Y_Testing == 0, :][:, selected_features]
                #BBB = X_Testing[Y_Testing == 1, :][:, selected_features]
                #print(AAA.shape)

                """
                P1 = X_Testing[Y_Testing == 0, :][:, 0][:, selected_features]
                N1 = X_Testing[Y_Testing == 1, :][:, 0][:, selected_features]

                # plt.plot(X1[filter(lambda a:a<len(P0_Index),X1),:],Y1[filter(lambda a:a<len(P0_Index),X1),:],'b')
                plt.plot(N0_Index[window_size - 1:], N1, 'b')
                P0_Index.pop(-1)
                plt.xlim(window_size, len(X0))
                plt.plot(P0_Index, P1, 'r')
                plt.ylim(-0.1, 0.4)
                print(str(time_scale_size)+" is completed!")
                """



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
        if  not eachfile=='HB_Nimda.txt':continue
        if '.py' in eachfile or '.DS_' in eachfile: continue
        if '.txt' in eachfile:
            pass
        else:
            continue
        print(eachfile+ " is processing---------------------------------------------------------------------------------------------LSTM_Size is "+str(lstm_size_list[-1]))
        for lstm_size in lstm_size_list:
            for window_size_label in window_size_label_list:
                if window_size_label == 'true':
                    Method_Dict={"LSTM": 0}
                    #Method_Dict = {"LSTM": 0,"AdaBoost": 1, "DT": 2, "SVM": 3, "LOGREG": 4, "NB": 5}
                    #Method_Dict = {"LSTM": 0,"AdaBoost": 1, "SVM": 3, "NB": 5}

                    for window_size in window_size_list:
                        time_scale_size_list = get_all_subfactors(window_size)
                        #output_data_path = os.path.join(os.getcwd(),'NNNNNBBBBB_'+str(Noise_Ratio*100)+'%'+'SSSingle_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        #output_data_path = os.path.join(os.getcwd(),'NNNNNBBBBB_'+'MMMulti_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        #output_data_path = os.path.join(os.getcwd(),'Noise_'+str(Noise_Ratio*100)+'%'+'SSSingle_Traditional' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)
                        output_data_path = os.path.join(os.getcwd(),'Half_Minutes_Window_Size_' + str(window_size) + '_LS_' + str(lstm_size)+Normalization_Method)

                        if not os.path.isdir(output_data_path):
                            os.makedirs(output_data_path)
                        time_scale_size_list = [-1,0,2,3,4,5,10]
                        plt.figure()
                        for time_scale_size in time_scale_size_list:
                            print("TIME SCALE IS " + str(time_scale_size))
                            if  time_scale_size == 0:
                                plt.subplot(3,2,1)
                            if  time_scale_size == 2:
                                plt.subplot(3,2,2)
                            if  time_scale_size == 3:
                                plt.subplot(3,2,3)
                            if  time_scale_size == 4:
                                plt.subplot(3,2,4)
                            if  time_scale_size == 5:
                                plt.subplot(3,2,5)
                            if  time_scale_size == 10:
                                plt.subplot(3,2,6)
                            else:
                                pass
                            Main(Bin_or_Multi_Label,Method_Dict,eachfile,window_size_label,lstm_size,Noise_Ratio,window_size,time_scale_size)
                        plt.savefig("ABC.png",dpi=200)
                        plt.show()
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

