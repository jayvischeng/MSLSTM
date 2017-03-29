import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
import sklearn
import loaddata
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




def evaluation(label,predict,trigger_flag,evalua_flag,positive_label=1,negative_label=0):

    try:
        label_ = loaddata.reverse_one_hot(label)
        predict_ = loaddata.reverse_one_hot(predict)
    except:
        label_ = list(label)
        predict_ = list(predict)

    ac_positive = 0
    ac_negative = 0
    correct = 0
    error_rate_flag = False


    #output class error rate
    predict_positive = predict_.count(positive_label)
    total_positive = label_.count(positive_label)


    predict_negative = predict_.count(negative_label)
    total_negative = label_.count(negative_label)
    print("TP is "+str(total_positive))
    print("TN is "+str(total_negative))

    for tab_sample in range(len(predict_)):
        if predict_[tab_sample] == label_[tab_sample]:
            correct += 1
            if int(label_[tab_sample]) == positive_label:
                ac_positive += 1
            elif int(label_[tab_sample]) == negative_label:
                ac_negative += 1


    if evalua_flag == True:
        try:
            ACC_A = float(ac_positive) / predict_positive
        except:
            ACC_A = float(ac_positive) * 100 / (predict_positive + 1)

        AUC = roc_auc_score(label_, np.transpose(np.array(predict_)))
        G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive))

        PRECISION = ACC_A
        RECALL = float(ac_positive) / total_positive
        ACCURACY = round(float(ac_positive + ac_negative) / len(label_), 5)
        #F1_SCORE = sklearn.metrics.f1_score(label_, predict_)
        try:
            F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
        except:
            F1_SCORE = 0.01 * round((2 * PRECISION * RECALL) / (PRECISION + RECALL + 1), 5)

        return {"ACCURACY": ACCURACY, "F1_SCORE": F1_SCORE, "AUC": AUC, "G_MEAN": G_MEAN}


    else:
        AUC = roc_auc_score(label_, np.transpose(np.array(predict_)))
        G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive))
        FPR,TPR,AUC = ComputeAUC(label_,predict_)
        print({"FPR":FPR,"TPR":TPR,"AUC":AUC,"G_MEAN":G_MEAN})
        return {"FPR":FPR,"TPR":TPR,"AUC":AUC,"G_MEAN":G_MEAN}
        #return fpr,tpr,auc#evaluate auc
def evaluation2(label,predict,trigger_flag,evalua_flag,positive_label=1,negative_label=0):

    if trigger_flag or evalua_flag == False:
        Output_Class = [[] for i in range(len(label[0]))]
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
    else:
        Output_Class = []
        for tab_sample in range(len(predict)):
            Output_Class.append(int(predict[tab_sample]))

    ac_positive = 0
    ac_negative = 0
    correct = 0
    error_rate_flag = False
    if trigger_flag or evalua_flag == False:
        for tab_class in range(len(Output_Class)):
            error_rate_flag = True
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
    else:
        #output class error rate
        predict_positive = Output_Class.count(positive_label)
        total_positive = list(label).count(positive_label)

        predict_negative = Output_Class.count(negative_label)
        total_negative = list(label).count(negative_label)
        for tab_sample in range(len(predict)):
            if predict[tab_sample] == label[tab_sample]:
                correct += 1
                if int(label[tab_sample]) == positive_label:
                    ac_positive += 1
                elif int(label[tab_sample]) == negative_label:
                    ac_negative += 1

    #if error_rate_flag == True:
        #print("Error Rate is :"+str((len(Output_Class) - correct)/float(len(Output_Class))))
        #return {"Error_Rate":(len(Output_Class) - correct)/float(len(Output_Class))}

    #PlottingAUC(ReverseEncoder(label),ReverseEncoder(np.transpose(np.array(Output_Class))))
    if evalua_flag == True:
        try:
            ACC_A = float(ac_positive) / predict_positive
        except:
            ACC_A = float(ac_positive) * 100 / (predict_positive + 1)

        AUC = roc_auc_score(label, np.transpose(np.array(Output_Class)))
        G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive))

        PRECISION = ACC_A
        RECALL = float(ac_positive) / total_positive
        ACCURACY = round(float(ac_positive + ac_negative) / len(label), 5)
        #F1_SCORE = sklearn.metrics.f1_score(label, Output_Class)
        try:
            F1_SCORE = round((2 * PRECISION * RECALL) / (PRECISION + RECALL), 5)
        except:
            F1_SCORE = 0.01 * round((2 * PRECISION * RECALL) / (PRECISION + RECALL + 1), 5)

        return {"ACCURACY": ACCURACY, "F1_SCORE": F1_SCORE, "AUC": AUC, "G_MEAN": G_MEAN}


    else:
        AUC = roc_auc_score(label, np.transpose(np.array(Output_Class)))
        G_MEAN = np.sqrt(float(ac_positive * ac_negative) / (total_negative * total_positive))
        FPR,TPR,AUC = ComputeAUC(label,predict)
        print({"FPR":FPR,"TPR":TPR,"AUC":AUC,"G_MEAN":G_MEAN})
        return {"FPR":FPR,"TPR":TPR,"AUC":AUC,"G_MEAN":G_MEAN}
        #return fpr,tpr,auc#evaluate auc




