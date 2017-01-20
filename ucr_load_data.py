import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import loaddata
def return_index(X,K):
    mylist = []
    mylist.append(0)
    for tab in range(1,len(X)-1):
        delta1 = X[tab] - X[tab - 1]
        delta2 = X[tab + 1] - X[tab]
        if delta1 == 0 and delta2 == 0: pass
        elif delta1*delta2 <=0:mylist.append(tab)
        else:
            if ((-1<delta1<0) and (delta2/delta1)>K) or ((delta1>=1) and (delta2/delta1)>K):
                mylist.append(tab)
            elif ((0<delta1<1) and (delta2/delta1)<(1/K)) or ((delta1<=-1) and (delta2/delta1)<(1/K)):
                mylist.append(tab)
    #print(len(mylist))
    return mylist

def return_merged_index(trainX,testX,K=100):
    trainX_2 = []
    for each in trainX:
        temp = each[return_index(each,K)]
        temp2 = np.pad(temp,pad_width=900-len(temp),mode='constant',constant_values=0)[900-len(temp):]
        trainX_2.append(temp2)
    testX_2 = []
    for each in testX:
        #testX_2.append(each[return_index(each,K)])
        temp = each[return_index(each,K)]
        temp2 = np.pad(temp,pad_width=900-len(temp),mode='constant',constant_values=0)[900-len(temp):]
        testX_2.append(temp2)
    #A1 = list(set(A1))
    #A2 = list(set(A2))
    #A = A1[:]
    #for tab in range(len(A2)):
        #if not A2[tab] in A:
            #A.append(A2[tab])

    #return A
    print("AAAAAAAAAAAAAAAAAA")
    trainX_2 = np.array(trainX_2)
    testX_2 = np.array(testX_2)
    print(trainX_2.shape)
    print(testX_2.shape)
    print(len(trainX_2[0]))
    print(len(trainX_2[-1]))
    print("BBBBBBBBBBBBBBBB")
    return np.array(trainX_2),np.array(testX_2)

def load_ucr_data(is_multi_scale=False,filename='wafer'):
    input_folder = os.path.join(os.getcwd(),'UCR_TS_Archive_2015')
    output_folder = os.path.join(os.getcwd(),'UCR_TS_Archive_2015_OutPut')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    filelist=os.listdir(input_folder)
    print(filelist)
    count = 0
    for each in filelist:
        if each == filename: pass
        else:continue
        print(each)
        train_file = os.path.join(os.path.join(input_folder,each),each+'_TRAIN')
        test_file = os.path.join(os.path.join(input_folder,each),each+'_TEST')
        trainX = []
        trainY= []
        with open(os.path.join(input_folder,train_file))as fin:
            for eachline in fin.readlines():
                val = eachline.split(',')
                trainY.append(int(val[0]))
                trainX.append(list(map(lambda a:float(a),val[1:])))
        testX = []
        testY= []
        with open(os.path.join(input_folder,test_file))as fin:
            for eachline in fin.readlines():
                val = eachline.split(',')
                testY.append(int(val[0]))
                testX.append(list(map(lambda a:float(a),val[1:])))
        count += 1
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    print("TEMP IS ")
    print(trainX.shape)
    print(trainY[0])

    #merged_index = return_merged_index(trainX, testX)

    trainY_encoder = LabelEncoder()
    trainY_encoder.fit(trainY)
    testY_encoder = LabelEncoder()
    testY_encoder.fit(testY)

    trainY = np_utils.to_categorical(trainY_encoder.transform(trainY))
    testY = np_utils.to_categorical(testY_encoder.transform(testY))

    if is_multi_scale == True:
        Scale_Level = 6
        Wave_Type = 'db1'
        trainX_M2 = []
        testX_M2 = []

        trainX_M, trainX, trainY = loaddata.Multi_Scale_Wavelet(trainX, trainY, Scale_Level, True, Wave_Type)
        testX_M, testX, testY = loaddata.Multi_Scale_Wavelet(testX, testY, Scale_Level, True, Wave_Type)

        # Plot the different scale
        # Multi_Scale_Plotting(Data_Multi_Level_X,Data_X)

        for tab_level in range(Scale_Level):
            trainX_M2.append(np.reshape(trainX_M[tab_level],(len(trainX),len(trainX[0]),1)))
            testX_M2.append(np.reshape(testX_M[tab_level],(len(testX),len(testX[0]),1)))
        trainX_M2 = np.array(np.array(trainX_M2).transpose((1, 0, 2, 3)))  # batch_size, scale_levels, sequence_window, input_dim
        testX_M2 = np.array(np.array(testX_M2).transpose((1, 0, 2, 3))) # batch_size, scale_levels, sequence_window, input_dim

        print(trainX_M2.shape)
        print(len(testX_M2.shape))
        return trainX_M2,trainY,testX_M2,testY
    #trainX2 = np.reshape(trainX[:,merged_index],(len(trainX),len(merged_index),1))
    #testX2 = np.reshape(testX[:,merged_index],(len(testX),len(merged_index),1))
    #trainX2,testX2 = return_merged_index(trainX, testX)

    #trainX2 = np.reshape(trainX2,(len(trainX2),len(trainX2[0]),1))
    #testX2 = np.reshape(testX2,(len(testX2),len(testX2[0]),1))

    #trainX = np.reshape(trainX,(len(trainX),len(trainX[0]),1))
    #testX = np.reshape(testX,(len(testX),len(testX[0]),1))

    return trainX,trainY,testX,testY

#x_train,y_train,x_test,y_test = load_ucr_data(False,'Worms')
#print(x_train.shape[1])
#print(y_train.shape[1])
