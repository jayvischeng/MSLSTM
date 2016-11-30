import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

True =     np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]])
Predict = np.array([[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1]])

def Evaluation_WithoutS(True, Predict, positive_label=0, negative_label=1):
    ac_positive = 0
    ac_negative = 0
    #print(list(Predict))
    Predict2 = map(lambda a:int(a),list(Predict))
    # print("In TRUE CLASS1:"+str(list(np.transpose(True)[0]).count(0)))
    # print("In TRUE CLASS1:"+str(list(np.transpose(True)[0]).count(1)))

    # print("In TRUE CLASS2:"+str(list(np.transpose(True)[1]).count(0)))
    # print("In TRUE CLASS2:"+str(list(np.transpose(True)[1]).count(1)))

    # print("In PREDICT CLASS1:"+str(Output_Class[0].count(0)))
    # print("In PREDICT CLASS1:"+str(Output_Class[0].count(1)))

    # print("In PREDICT CLASS2:"+str(Output_Class[1].count(0)))
    # print("In PREDICT CLASS2:"+str(Output_Class[1].count(1)))
    predict_negative = list(Predict2).count(negative_label)
    total_negative = list(True).count(negative_label)
    predict_positive = list(Predict2).count(positive_label)
    total_positive = list(True).count(positive_label)
    for tab in range(len(Predict2)):
        if Predict2[tab] == True[tab] and True[tab] == negative_label:
            ac_negative += 1
        elif Predict2[tab] == True[tab] and True[tab] == positive_label:
            ac_positive += 1

    try:
        ACC_R = float(ac_negative) / predict_negative
    except:
        ACC_R = float(ac_negative) * 100 / (predict_negative + 1)
    try:
        ACC_A = float(ac_positive) / predict_positive
    except:
        ACC_A = float(ac_positive) * 100 / (predict_positive + 1)

    Y_Encoder = LabelEncoder()
    Y_Encoder.fit(True)
    encoded_True = Y_Encoder.transform(True)
    encoded_True = np_utils.to_categorical(encoded_True)

    Y_Encoder = LabelEncoder()
    Y_Encoder.fit(Predict)
    encoded_Predict = Y_Encoder.transform(Predict)
    encoded_Predict = np_utils.to_categorical(encoded_Predict)

    print(encoded_True)
    print(encoded_Predict)
    #PlottingAUC(encoded_True,encoded_Predict)
    AUC = roc_auc_score(True, Predict)
    G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive))

    PRECISION = ACC_A
    RECALL = float(ac_positive) / total_positive
    ACCURACY = round(float(ac_positive + ac_negative) / len(True), 5)
    try:
        F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
    except:
        F1_SCORE = 0.01 * round((2 * PRECISION * RECALL) / (PRECISION + RECALL + 1), 5)

    return {"ACCURACY": ACCURACY, "F1_SCORE": F1_SCORE, "AUC": AUC, "G_MEAN": G_MEAN}


# Evaluation(np.array(True),np.array(Predict))


def ComputeAUC(y_test,y_predict):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
        #print(y_test[:, i])
       # print("BBBBBBBBBBBBBBBBBBBBBBBBBB")
        #print(y_predict[:, i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot of a ROC curve for a specific class
    #return (fpr[0]+fpr[1])/2,(tpr[0]+tpr[1])/2,(roc_auc[0]+roc_auc[1])/2


    return fpr[0],tpr[0],roc_auc[0]



def ReverseEncoder(y):
    temp = []
    for tab1 in range(len(y)):
            temp.append(list(y[tab1]).index(max(list(y[tab1]))))
    return temp


def Evaluation(True,Predict,positive_label=0,negative_label=1):
    StateFalseAlarmOutPut = []
    try:
        Output_Class = [[] for i in range(len(True[0]))]
    except:
        Output_Class = [[] for i in range(len(True))]
    if len(Predict) == len(True):
        for tab_sample in range(len(True)):
            max_index = Predict[tab_sample].argmax(axis=0)
            for tab_class in range(len(Output_Class)):
                if tab_class == max_index:
                    Output_Class[tab_class].append(1)
                else:
                    Output_Class[tab_class].append(0)
    else:
        print("Error!")

    ac_positive = 0
    ac_negative = 0
    #print("In TRUE CLASS1:"+str(list(np.transpose(True)[0]).count(0)))
    #print("In TRUE CLASS1:"+str(list(np.transpose(True)[0]).count(1)))

    #print("In TRUE CLASS2:"+str(list(np.transpose(True)[1]).count(0)))
    #print("In TRUE CLASS2:"+str(list(np.transpose(True)[1]).count(1)))

    #print("In PREDICT CLASS1:"+str(Output_Class[0].count(0)))
    #print("In PREDICT CLASS1:"+str(Output_Class[0].count(1)))

    #print("In PREDICT CLASS2:"+str(Output_Class[1].count(0)))
    #print("In PREDICT CLASS2:"+str(Output_Class[1].count(1)))

    for tab_class in range(len(Output_Class)):
        if True[0][tab_class] == negative_label:
            predict_negative = Output_Class[tab_class].count(1)
            total_negative = list(np.transpose(True)[tab_class]).count(1)
            #print("The CLASS is -------------"+str(tab_class))
            #print("predict_negative is " + str(predict_negative))
            #print("total_negative is " + str(total_negative))

            for tab_sample in range(len(True)):
                if Output_Class[tab_class][tab_sample]==1 and Output_Class[tab_class][tab_sample] == int(True[tab_sample][tab_class]) :
                    ac_negative += 1

        elif True[0][tab_class] == positive_label:
            #print(tab_class)
            predict_positive = Output_Class[tab_class].count(1)
            total_positive = list(np.transpose(True)[tab_class]).count(1)
            #print("The CLASS is -------------"+str(tab_class))
            #print("predict_positive is " + str(predict_positive))
            #print("total_positive is " + str(total_positive))
            for tab_sample in range(len(True)):
                if Output_Class[tab_class][tab_sample]==1 and Output_Class[tab_class][tab_sample] == int(True[tab_sample][tab_class]):
                    ac_positive += 1


    #print(np.transpose(True))
    #print(Output_Class)
    Predict2 = np.transpose(Predict)
    for tab in range(len(True)):
        if Output_Class[0][tab] == 0 and Output_Class[1][tab]!=1:
            print("111111111111111111111111111111111111111111")
            print(tab)
            print(Predict2[0][tab])
            print(Predict2[1][tab])
        if Output_Class[0][tab] == 1 and Output_Class[1][tab]!=0:
            print("222222222222222222222222222222222222222222")
            print(tab)
            print(Predict2[0][tab])
            print(Predict2[1][tab])
    #print("ac_negative is "+str(ac_negative))
    #print("ac_positive is "+str(ac_positive))
    #print("predict_negative is "+str(predict_negative))
    #print("predict_positive is "+str(predict_positive))
    #print("total_negative is "+str(total_negative))
    #print("total_positive is "+str(total_positive))
    #print("True is "+str(len(True)))
    #if tab_cross == 0:
        # print('Window Size is ' + str(window_size) + " Time Scale is " + str(time_scale_size)+"Tesing Length is :"+str(len(True)))
        #with open("CrossTab_" + str(tab_cross) + "_" + str(eachMethod) + ".txt", "w")as fout:
            #for tab in range(len(True)):
                #fout.write(str(Output[tab]) + '\t\t' + str(True[tab]) + '\n')
    try:
        ACC_R = float(ac_negative) / predict_negative
    except:
        ACC_R = float(ac_negative) * 100 / (predict_negative + 1)
    try:
        ACC_A = float(ac_positive) / predict_positive
    except:
        ACC_A = float(ac_positive) * 100 / (predict_positive + 1)

    #print(Predict[:,0])
    #print(np.transpose(Predict)[:,0])
    #PlottingAUC(ReverseEncoder(True),ReverseEncoder(np.transpose(np.array(Output_Class))))
    fpr,tpr,auc = ComputeAUC(True,Predict)
    AUC = roc_auc_score(True,np.transpose(np.array(Output_Class)))
    G_MEAN=np.sqrt(float(ac_positive*ac_negative)/(total_negative*total_positive))

    PRECISION = ACC_A
    RECALL = float(ac_positive) / total_positive
    ACCURACY = round(float(ac_positive + ac_negative) / len(True), 5)
    try:
        F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
    except:
        F1_SCORE = 0.01 * round((2 * PRECISION * RECALL) / (PRECISION + RECALL + 1), 5)


    return {"ACCURACY":ACCURACY,"F1_SCORE":F1_SCORE,"AUC":AUC,"G_MEAN":G_MEAN}
    #return fpr,tpr,auc


#Evaluation(np.array(True),np.array(Predict))
#fp,tp,auc = Evaluation_WithoutS(True,Predict)


