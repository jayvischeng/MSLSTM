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