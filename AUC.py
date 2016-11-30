import c
import os
import da
import db2
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

Parameters = {}
# Parameters
#global filepath, filename, sequence_window, number_class, hidden_units, input_dim, learning_rate, epoch, is_multi_scale, training_level, cross_cv
# ---------------------------Fixed Parameters------------------------------
Parameters["filepath"] = os.getcwd()
Parameters["sequence_window"] = 10
Parameters["hidden_units"] = 200
Parameters["input_dim"] = 33
Parameters["number_class"] = 2
Parameters["cross_cv"] = 2
Parameters["fixed_seed_num"] = 1337
# -------------------------------------------------------------------------
Parameters["learning_rate"] = 0.001
Parameters["epoch"] = 250
Parameters["is_multi_scale"] = True
Parameters["training_level"] = 10
Parameters["is_add_noise"] = False
Parameters["noise_ratio"] = 0
Parameters["pooling_type"] = "diag pooling"
def PlotAUC(method_list,Parameters):
    for tab in range(len(method_list)):
        try:
            #print(method_list[tab] + " of C is processing----------------->>>>>>>>>>>>>>>>>>>>>")
            fpr[tab], tpr[tab], auc[tab] = c.Model(method_list[tab], Parameters)
            print(method_list[tab]+ " of C is completed++++++++++++++++<<<<<<<<<<<<<<<<<<<<<")
        except:
            try:
                #print(method_list[tab] + " of DA is processing----------------->>>>>>>>>>>>>>>>>>>>>")
                fpr[tab], tpr[tab], auc[tab] = da.Model(method_list[tab], Parameters)
                print(method_list[tab] + " of DA is completed++++++++++++++++<<<<<<<<<<<<<<<<<<<<<")

            except:
                #print(method_list[tab] + " of DB2 is processing----------------->>>>>>>>>>>>>>>>>>>>>")
                #fpr[tab], tpr[tab], auc[tab] = db2.Model(method_list[tab], Parameters)
                #print(method_list[tab] + " of DB2 is completed++++++++++++++++<<<<<<<<<<<<<<<<<<<<<")
                pass
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
    plt.savefig(eachfile + "_AUC.png", dpi=800)
    plt.savefig(eachfile + "_AUC.pdf", dpi=800)
for eachfile in filename_list:
    if not "Slammer" in eachfile:
        continue
    else:
        Parameters["filename"] = eachfile
        Parameters["learning_rate"] = 0.001
        Parameters["epoch"] = 75

        PlotAUC(method_list,Parameters)


