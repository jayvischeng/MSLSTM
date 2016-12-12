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
    plt.savefig(eachfile + "_AUC.png", dpi=800)
    plt.savefig(eachfile + "_AUC.pdf", dpi=800)


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



