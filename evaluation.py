import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
def ComputeAUC(y_test,y_predict):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot of a ROC curve for a specific class
    #return (fpr[0]+fpr[1])/2,(tpr[0]+tpr[1])/2,(roc_auc[0]+roc_auc[1])/2
    return fpr[0],tpr[0],roc_auc[0]



def ReverseEncoder(y):
    temp = []
    for tab1 in range(len(y)):
            temp.append(list(y[tab1]).index(max(list(y[tab1]))))
    return temp


def evaluation(label,predict,positive_label=0,negative_label=1):
    try:
        Output_Class = [[] for i in range(len(label[0]))]
    except:
        Output_Class = [[] for i in range(len(label))]
    if len(predict) == len(label):
        for tab_sample in range(len(label)):
            max_index = predict[tab_sample].argmax(axis=0)
            for tab_class in range(len(Output_Class)):
                if tab_class == max_index:
                    Output_Class[tab_class].append(1)
                else:
                    Output_Class[tab_class].append(0)
    else:
        print("Error!")

    ac_positive = 0
    ac_negative = 0

    for tab_class in range(len(Output_Class)):
        if label[0][tab_class] == negative_label:
            predict_negative = Output_Class[tab_class].count(1)
            total_negative = list(np.transpose(label)[tab_class]).count(1)

            for tab_sample in range(len(label)):
                if Output_Class[tab_class][tab_sample]==1 and Output_Class[tab_class][tab_sample] == int(label[tab_sample][tab_class]) :
                    ac_negative += 1
        elif label[0][tab_class] == positive_label:
            predict_positive = Output_Class[tab_class].count(1)
            total_positive = list(np.transpose(label)[tab_class]).count(1)
            for tab_sample in range(len(label)):
                if Output_Class[tab_class][tab_sample]==1 and Output_Class[tab_class][tab_sample] == int(label[tab_sample][tab_class]):
                    ac_positive += 1

    #try:
        #ACC_R = float(ac_negative) / predict_negative
    #except:
        #ACC_R = float(ac_negative) * 100 / (predict_negative + 1)
    try:
        ACC_A = float(ac_positive) / predict_positive
    except:
        ACC_A = float(ac_positive) * 100 / (predict_positive + 1)

    #PlottingAUC(ReverseEncoder(label),ReverseEncoder(np.transpose(np.array(Output_Class))))
    fpr,tpr,auc = ComputeAUC(label,predict)
    #AUC = auc
    AUC = roc_auc_score(label,np.transpose(np.array(Output_Class)))
    G_MEAN=np.sqrt(float(ac_positive*ac_negative)/(total_negative*total_positive))

    PRECISION = ACC_A
    RECALL = float(ac_positive) / total_positive
    ACCURACY = round(float(ac_positive + ac_negative) / len(label), 5)
    try:
        F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
    except:
        F1_SCORE = 0.01 * round((2 * PRECISION * RECALL) / (PRECISION + RECALL + 1), 5)


    return {"ACCURACY":ACCURACY,"F1_SCORE":F1_SCORE,"AUC":AUC,"G_MEAN":G_MEAN}
    #return fpr,tpr,auc#evaluate auc




