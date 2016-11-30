import os
import random
import numpy as np
import collections
import matplotlib.pyplot as plt
def get_all_subfactors(number):
    temp_list = []
    temp_list.append(1)
    temp_list.append(2)
    for i in range(3,number):
        if number%i == 0 :
            temp_list.append(i)
    temp_list.append(number)
    return temp_list
def Main(filename,window_size_list,lstm_size_list,Evaluation_List):
    global Evaluation_Dict_Max_Time_Scale,Normalization_Method
    Evaluation_Dict = collections.defaultdict(dict)
    Evaluation_DictOutPut = collections.defaultdict(dict)
    Evaluation_Dict_Max_Time_Scale = collections.defaultdict(dict)
    Evaluation_DictOutPut222 = collections.defaultdict(dict)
    Evaluation_Dict_Max_Time_Scale222 = collections.defaultdict(dict)

    for each_eval in Evaluation_List:
        Evaluation_DictOutPut222[each_eval] = collections.defaultdict(list)
    for each_eval in Evaluation_List:
        Evaluation_Dict_Max_Time_Scale222[each_eval] = collections.defaultdict(list)

    for each_eval in Evaluation_List:
        Evaluation_DictOutPut[each_eval] = collections.defaultdict(list)
    for each_eval in Evaluation_List:
        Evaluation_Dict_Max_Time_Scale[each_eval] = collections.defaultdict(list)

    Method_Dict = collections.defaultdict(list)
    global Method_list
    #Method_list = ["LSTM","AdaBoost","NB","DT","SVM","LOGREG"]

    for lstm_size in lstm_size_list:


        for window_size in window_size_list:
            for each_eval in Evaluation_List:
                Evaluation_Dict[each_eval] = collections.defaultdict(list)

            time_scale_size_list = get_all_subfactors(window_size)
            processingfolder = "Half_Minutes_Window_Size_"+str(window_size)+'_LS_'+str(lstm_size)+Normalization_Method
            filelist = os.listdir(processingfolder)

            for each_eval in Evaluation_List:
                for eachfile in filelist:
                    if "Bagging" in eachfile or "SubFeature" in eachfile:continue
                    if filename in eachfile and each_eval in eachfile:

                        for each_method in Method_list:
                            Method_Dict[each_method] = []
                        pass

                    else:continue
                    print(os.path.join(processingfolder,eachfile))
                    try:
                        with open(os.path.join(processingfolder,eachfile)) as fin:
                            vallines = fin.readlines()
                            for tab in range(len(vallines)):
                                if '(' in vallines[tab]:continue
                                temp_method = vallines[tab].split(':')[0].strip()
                                temp_value = float((vallines[tab].split(':')[-1].replace(",","")).strip())
                                #print(vallines[tab])
                                #print(temp_value)
                                if temp_value < 1:
                                    Evaluation_Dict[each_eval][temp_method].append(temp_value*100)
                                else:
                                    Evaluation_Dict[each_eval][temp_method].append(temp_value)

                    except:
                        pass

            for each_eval in Evaluation_List:
                for eachMethod in Method_list:
                    temp_list = Evaluation_Dict[each_eval][eachMethod]
                    try:
                        Evaluation_DictOutPut[each_eval][eachMethod].append(max(temp_list))
                        Evaluation_Dict_Max_Time_Scale[each_eval][eachMethod].append(time_scale_size_list[temp_list.index(max(temp_list))])
                    except:
                        Evaluation_DictOutPut[each_eval][eachMethod].append(0)
                        Evaluation_Dict_Max_Time_Scale[each_eval][eachMethod].append(0)
    for each_eval in Evaluation_List:
        for eachMethod in Method_list:
            value_list = Evaluation_DictOutPut[each_eval][eachMethod]
            value_list_time_scale = Evaluation_Dict_Max_Time_Scale[each_eval][eachMethod]
            for tab1 in range(len(window_size_list)):
                out_put_value_list = []
                out_put_value_list_time_scale = []
                for tab2 in range(len(lstm_size_list)):
                    out_put_value_list.append(value_list[tab1+len(window_size_list)*tab2])
                    out_put_value_list_time_scale.append(value_list_time_scale[tab1+len(window_size_list)*tab2])

                Evaluation_DictOutPut222[each_eval][eachMethod].append(max(out_put_value_list))
                Evaluation_Dict_Max_Time_Scale222[each_eval][eachMethod].append(out_put_value_list_time_scale[out_put_value_list.index(max(out_put_value_list))])

    return Evaluation_DictOutPut222,Evaluation_Dict_Max_Time_Scale222



if __name__=='__main__':
    global Normalization_Method
    Normalization_Method = "_L2norm"
    filename = "HB_AS_Leak_13237"
    marker_style = dict(linestyle=':', color='cornflowerblue', markersize=10)
    if filename == "B_Code_Red_I" or filename == "B_Slammer":
        window_size_list = [10,20,30,40,50,60]
        y_base = 60
    else:
        if filename == "B_C_N_S":
            y_upper = 95
        else:
            y_upper = 100
        window_size_list = [10,20,30,40,50,60]
        #window_size_list = [20,30,40,50,60]
        #[84.039,86.018, 88.206, 88.654, 89.17, 89.629]
        y_base = 60

    lstm_size_list = [100]
    Evaluation_List = ["ACC_R","ACC_A","ACC_L","Auc","G_mean","F1_score"]
    #Evaluation_List = ["ACC_L"]
    global Method_list
    #Method_list = ["LSTM","AdaBoost","NB","DT","SVM","LOGREG"]
    Method_list = ["LSTM","AdaBoost","NB","SVM"]

    #color_list = ['r','g','b','c','m','y']
    #color_dict = {"KNN":'c>',"AdaBoost":'r<',"DT":'yo',"LR":'m',"SVM":'g',"LSTM":'b'}

    #color_dict = {"KNN":'cx',"AdaBoost":'rs',"DT":'y>',"LR":'mo',"SVM":'gD',"LSTM":'b*'}
    #color_dict = {"NB":'cx',"AdaBoost":'r*',"DT":'y>',"LOGREG":'mo',"SVM":'gD',"LSTM":'bs'}
    color_dict = {"NB":'cx',"AdaBoost":'r*',"SVM":'gD',"LSTM":'bs'}

    #method_dict = {"NB":"NB","AdaBoost":"Ada.Boost","DT":"DT","LOGREG":"LOGREG","SVM":"SVM","LSTM":"TSLSTM"}
    method_dict = {"NB":"NB","AdaBoost":"Ada.Boost","SVM":"SVM","LSTM":"MS-LSTM"}

    #line_dict = {"KNN":':',"AdaBoost":'-.',"DT":'-.',"LR":'--',"SVM":':',"LSTM":'--'}
    #line_dict = {"NB":':',"AdaBoost":':',"DT":'--',"LOGREG":'--',"SVM":'-.',"LSTM":'-.'}
    line_dict = {"NB":':',"AdaBoost":':',"SVM":'-.',"LSTM":'-.'}

    Evaluation_Dict,Evaluation_Dict_Max_Time_Scale = Main(filename,window_size_list,lstm_size_list,Evaluation_List)

    if not os.path.isdir(os.path.join(os.getcwd(),"AAA_Images_WWW_"+Normalization_Method)):
        os.makedirs(os.path.join(os.getcwd(),"AAA_Images_WWW_"+Normalization_Method))


    for each_evalk,each_evalv in Evaluation_Dict.items():
        print("-------------------------------------------")
        title = each_evalk
        X = window_size_list
        #Y_list = [[] for i in range(len(each_evalv))]
        plt.figure()
        signal_list = [0,1]
        count = 0
        for eachMethod,eachList in each_evalv.items():
            #plt.subplot(1,)
            Y = eachList
            Y_max_time_scale = Evaluation_Dict_Max_Time_Scale[each_evalk][eachMethod]
            print("For "+eachMethod+": the max "+each_evalk+ " is "+str(round(np.max(Y),1)))
            #print(X)
            #print(Y_max_time_scale)
            #print(X)
            #if eachMethod=="NB":
                #Y=[84.039, 86.018, 88.206, 88.654, 89.17, 89.629]
            #print(Y)
            plt.plot(X,Y,color_dict[eachMethod]+line_dict[eachMethod],linewidth=3,markersize=8,label=method_dict[eachMethod])
            for i, txt in enumerate(Y_max_time_scale):
                plt.annotate('('+str(txt)+')', xy=(X[i],Y[i]),xycoords='data',xytext=(X[i],Y[i]),size = 10)


            plt.xlim(10,window_size_list[-1]+5)
            plt.ylim(y_base,100)
            plt.grid()
            #plt.tight_layout()
            legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc='center right', ncol=2, mode="expand", borderaxespad=0.)
                                #shadow=True, fontsize='x-small')
            #legend.get_frame().set_facecolor('#00FFCC')
            count += 1
        #print("start"),
        if each_evalk=="ACC_R":
            plt.ylabel("Regular precision")
        elif each_evalk=="ACC_A":
            plt.ylabel("Anomaly precision")
        elif each_evalk=="ACC_L":
            plt.ylabel("Accuracy")
        elif each_evalk=="F1_score":
            plt.ylabel("F-score")
        else:
            plt.ylabel(each_evalk)
        plt.xlabel('Window Size')
        #plt.show()

        plt.savefig(os.path.join(os.path.join(os.getcwd(),"AAA_Images_WWW_"+Normalization_Method),filename+'_'+title+".pdf"),dpi=400)
        #image = plt.imread(os.path.join(os.path.join(os.getcwd(),"Images_W_"+Normalization_Method),filename+'_'+title+".png"))
        #arr = np.asarray(image)
        #plt.imshow(arr, cmap='gray')
        #plt.show()

#import matplotlib.rcsetup as rcsetup
#print(rcsetup.all_backends)
#import matplotlib
#print(matplotlib.matplotlib_fname())