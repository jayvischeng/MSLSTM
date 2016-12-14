import matplotlib.pyplot as plt
import matplotlib
import os
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()
def MC_Plotting(Data,row,col,x_label='x_label',y_label='y_label',suptitle='super_title',save_fig='save_fig'):
    X = [i+1 for i in range(len(Data[0]))]

    plt.figure(figsize=(row*col*4,col))
    for tab in range(row*col):
        plt.subplot(row,col,tab+1)
        plt.plot(X,Data[tab],'s-')
        plt.xlabel(x_label,fontsize=10)
        plt.ylabel(y_label,fontsize=10)
        plt.ylim(40,100)
        plt.grid()
        plt.tick_params(labelsize=10)
    plt.tight_layout()
    plt.suptitle(suptitle)
    plt.savefig(save_fig,dpi=200)
    plt.show()

A1 = [51.5,54.2,55.1,55.4,55.8,57.3,49.6,52.2,63.4,63.5]#"AS_LEAK"
A2 = [50.6,53.9,55.3,54.3,56.8,52.7,54.7,52.1,63.3,63.3]
A3 = [50.4,54.5,55.7,54.3,56.1,51.0,54.6,51.9,63.3,63.3]
A = []
A.append(A1)
A.append(A2)
A.append(A3)

MC_Plotting(A,1,3)

import matplotlib
import matplotlib.pyplot as plt
def set_style():
    plt.style.use(['classic'])
    matplotlib.rc("font", family="serif")
set_style()

filename_list = ["HB_AS_Leak.txt","HB_Slammer.txt","HB_Nimda.txt","HB_Code_Red_I.txt"]
method_list = ["SVM","NB","DT","Ada.Boost","MLP","RNN","LSTM","MS-LSTM"]
fpr = [0 for i in range(len(method_list))]
tpr = [0 for i in range(len(method_list))]
auc = [0 for i in range(len(method_list))]


def PlotAUC(method_list,Parameters):
    plt.figure()
    #color_list = ['y', 'g', '#FF8C00', 'c', 'b', 'r', 'm']
    color_list = ['y', 'g','#FF8C00','#FD8CD0','c', 'b', 'r', 'm']
    for tab in range(len(method_list)):
        plt.plot(fpr[tab], tpr[tab], color_list[tab], label=method_list[tab] + ' ROC curve (area = %0.2f)' % auc[tab])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.title('Receiver operating characteristic of '+Parameters['filename'].replace('HB_','').split('.')[0],fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.savefig("_AUC.png", dpi=800)
    plt.savefig("_AUC.pdf", dpi=800)


def Plotting(filename, subtitle, method):
    # Predict = []
    # True = []
    Temp = []
    with open(os.path.join(os.getcwd(), filename))as fin:
        for each in fin.readlines():
            Temp.append(int(each))
            # val = each.split('\t\t')
            # Predict.append(int(val[0].strip()))
            # True.append(int(float(val[-1].strip())))

    Temp = Temp[2340:4740]

    if "SVM" in filename:
        print(Temp)

    X = [i + 1 for i in range(len(Temp))]

    plt.xlim(0, len(X))
    plt.ylim(-0.5, 2.0)
    if not method == "Original":
        """
        if method == "NB":
            Temp = []
            for each in Predict:
                if each==1:
                    Temp.append(each)
                elif each==-1:
                    Temp.append(0)
            Predict = Temp
            print(Predict)
        print(len(X))
        print(len(Predict))
        """
        plt.plot(X, Temp, 'b.', markersize=2, label='Predict')
    else:
        plt.plot(X, Temp, 'r.', markersize=2, label='True')

    plt.legend(loc=1, fontsize=12)
    # plt.legend(bbox_to_anchor=(1, 1),
    # bbox_transform=plt.gcf().transFigure)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('(' + subtitle + ')' + "  " + method)
    x = len(X) / 2
    plt.xticks([1, 400, 800, 1200, 1600, 2000, 2500])
    # plt.grid()
    # plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.1)
    # plt.minorticks_on()
    plt.axvline(x, ymin=-1, ymax=2, linewidth=2, color='g')  # plt.title('Testing Sequence')
    # plt.axvline(x+30, ymin=-1, ymax=2, linewidth=2, color='g')    #plt.title('Testing Sequence')

    # plt.grid()


def PlotStat(filename, Method_List):
    subtitle = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    plt.figure(figsize=(12, 6), dpi=800)
    plt.subplot(3, 3, 1)
    filename_ = "StatFalseAlarm_" + filename + "_True.txt"
    Plotting(filename_, subtitle[0], "Original")

    for tab in range(len(Method_List)):
        filename_ = "StatFalseAlarm_" + filename + "_" + Method_List[tab] + "_" + "_Predict.txt"
        plt.subplot(3, 3, tab + 1)
        Plotting(filename_, subtitle[tab + 1], Method_List[tab])

    plt.tight_layout()
    plt.savefig("StateFalseAlarm_" + filename + ".png", dpi=400)
    plt.show()

"""
[0.023529412]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.019327732]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.01512605]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.018487396]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.0092436979]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.012605042]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.034453783]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.071428575]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.072268911]
I tensorflow/core/kernels/logging_ops.cc:79] this is output for output_correct_pred : [0.35630253]
"""
import os
import da
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#print(plt.style.available)
def set_style():
    plt.style.use(['classic'])
    matplotlib.rc("font", family="serif")
set_style()


#matplotlib.rcParams['backend'] = 'GTKCairo'
#Plotting different wavelet
Parameters = {}
# Parameters
#global filepath, filename, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv
# ---------------------------Fixed Parameters------------------------------
#Parameters["filename"] = "HB_AS_Leak.txt"
#Parameters["filename"] = "HB_Nimda.txt"
#Parameters["filename"] = "HB_Code_Red_I.txt"
#Parameters["filename"] = "HB_Slammer.txt"

Parameters["filepath"] = os.getcwd()
Parameters["sequence_window"] = 10
Parameters["hidden_units"] = 200
Parameters["input_dim"] = 33
Parameters["number_class"] = 2
Parameters["cross_cv"] = 2
Parameters["fixed_seed_num"] = 1337
Parameters["is_add_noise"] = False
Parameters["noise_ratio"] = 0
# -------------------------------------------------------------------------
Parameters["learning_rate"] = 0.001
Parameters["epoch"] = 150
Parameters["is_multi_scale"] = True
Parameters["training_level"] = 10
Parameters["pooling_type"] = "mean pooling"
wave_type = ["db1"]
#wave_type = ["db1","haar","bior1.1","rbio1.1"]
filename_list = ["HB_AS_Leak.txt","HB_Nimda.txt","HB_Slammer.txt","HB_Code_Red_I.txt"]
#filename_list = ["HB_Nimda.txt"]
lr = 0.0005
learning_rate = [lr,lr,lr,lr]
filename_list_label = {"HB_AS_Leak.txt":"AS Leak","HB_Nimda.txt":"Nimda","HB_Slammer.txt":"Slammer","HB_Code_Red_I.txt":"Code Red I"}

color_type = ['g-s','b-*','m-^','c-o']
#wave_type_name = ["Daubechies Filter","Haar","Biorthogonal","Reverse Biorthogonal"]
base = 20
filename_result = {}
filename_result2 = {}


for tab_filename in filename_list:
    Parameters["learning_rate"] = learning_rate[filename_list.index(tab_filename)]
    Parameters["filename"] = tab_filename
    Parameters["wave_type"] = "db1"
    #Parameters["batch_size"] = 2270
    filename_result[tab_filename] = []
    filename_result2[tab_filename] = []

    for tab_level in range(2,Parameters["sequence_window"]+1):
        Parameters["training_level"] = tab_level
        result = da.Model("MS-LSTM",Parameters)
        #train_loss,val_loss,train_acc,val_acc,weight_list,RESULT = da.Model("MS-LSTM")

        filename_result[tab_filename].append(result["ACCURACY"])
        filename_result2[tab_filename].append(result["F1_SCORE"])

#wave_type_db=[81.66,82.79,85.47,80.25,80.90,81.55,82.80,87.57,92.82,93.35,95.64]
#wave_type_haar=[81.66,82.79,85.47,80.25,80.90,81.55,82.80,87.57,92.82,93.35,95.64]
#wave_type_sym= []
with open("aaa.txt","a")as fout:
    for eachk,eachv in filename_result.items():
        fout.write(eachk)
        fout.write(''.join(eachv))
    for eachk,eachv in filename_result2.items():
        fout.write(eachk)
        fout.write(''.join(eachv))
print(filename_result)
print(filename_result2)

count = 0
for eachk,eachv in filename_result.items():
    X = [i+2 for i in range(len(eachv))]
    eachv = map(lambda a:100*a,eachv)
    print(eachv)
    plt.plot(X,eachv,color_type[count],label=filename_list_label[eachk])
    count += 1


plt.legend(loc="lower right",fontsize=12)
plt.ylabel("Accuracy",fontsize=12)
#if "AS" in Parameters["filename"]:
    #plt.ylim(35,75)
#else:
    #plt.ylim(50,100)
plt.ylim(base,100)
plt.tick_params(labelsize=12)
plt.xlabel("Scale Levels",fontsize=12)
plt.grid()
plt.savefig("Wave_Let_"+'_'+str(base)+'_'+str(lr)+'_'+".png",dpi=400)
plt.title("Wavelet Family: Daubechies/Filter Length: 2")
#plt.show()




