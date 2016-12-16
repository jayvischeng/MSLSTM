import matplotlib
matplotlib.use('GTKAgg')
#matplotlib.rcParams['backend'] = 'GTKCairo'
import os
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()
def epoch_acc_plotting(filename,case_list,sequence_window,corss_val_label,learning_rate,train_acc_list,val_acc_list):
    if not os.path.isdir(os.path.join(os.getcwd(),'picture')):
        os.makedirs(os.path.join(os.getcwd(),'picture'))
    epoch = len(train_acc_list[0])
    color_list = ['y', 'g','#FF8C00','#FD8CD0','c', 'b', 'r', 'm']
    #color_list = ['y', 'g', 'c', 'b', 'r', 'm']

    x = [i+1 for i in range(epoch)]
    plt.figure()
    for tab in range(len(case_list)):
        plt.plot(x,train_acc_list[tab],color_list[tab],label=case_list[tab] + ' train acc')
        plt.plot(x, val_acc_list[tab], color_list[len(case_list)+tab],label=case_list[tab] +' val acc')
    plt.xlabel('Epoch',fontsize=12)
    if 'AS' in filename:
        plt.ylim(0.0,1.0)
    else:
        plt.ylim(0.05,1.05)
    plt.ylabel('Accuracy',fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.legend(loc='lower right',fontsize=8)
    plt.title(filename.split('.')[0].replace('HB_','')+'/sw: '+str(sequence_window)+"/lr: "+str(learning_rate))
    if corss_val_label == 0:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Epoch_ACC_"+filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".pdf"),dpi=400)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Epoch_ACC_"+filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png"),dpi=400)
    else:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Epoch_ACC_" + filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".pdf"), dpi=400)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Epoch_ACC_" + filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png"), dpi=400)

def epoch_loss_plotting(filename,case_list,sequence_window,cross_val_label,learning_rate,train_loss_list,val_loss_list):
    if not os.path.isdir(os.path.join(os.getcwd(),'picture')):
        os.makedirs(os.path.join(os.getcwd(),'picture'))
    epoch = len(train_loss_list[0])
    color_list = ['y', 'g','#FF8C00','#FD8CD0','c', 'b', 'r', 'm']

    x = [i+1 for i in range(epoch)]
    plt.figure()
    for tab in range(len(case_list)):
        plt.plot(x,train_loss_list[tab],color_list[tab],label=case_list[tab] + ' train acc')
        plt.plot(x, val_loss_list[tab], color_list[len(case_list)+tab],label=case_list[tab] +' val acc')

    plt.xlabel('Epoch',fontsize=12)
    plt.ylim(0.5,0.95)
    plt.ylabel('Loss',fontsize=12)
    plt.grid()
    plt.tick_params(labelsize=12)
    plt.legend(loc='upper right',fontsize=8)
    plt.title(filename.split('.')[0].replace('HB_','')+'/sw: '+str(sequence_window)+"/lr: "+str(learning_rate))

    if cross_val_label == 0:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Epoch_Loss_"+filename+"_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".pdf"),dpi=400)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Epoch_Loss_"+filename+"_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png"),dpi=400)
    else:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Epoch_Loss_" + filename + "_SW_" + str(sequence_window)+"_LR_"+str(learning_rate) + ".pdf"), dpi=400)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Epoch_Loss_" + filename + "_SW_" + str(sequence_window)+"_LR_"+str(learning_rate) + ".png"), dpi=400)

def weight_plotting(filename,sequence_window,corss_val_label,learning_rate,weight_list):
    if not os.path.isdir(os.path.join(os.getcwd(),'picture')):
        os.makedirs(os.path.join(os.getcwd(),'picture'))
    weight_list_new = np.transpose(weight_list)
    #a,b,c = weight_list_new.shape
    subtitle = ['a', 'b', 'c', 'd', 'e', 'f','g','h','i','j']

    X = [i for i in range(len(weight_list_new[0][0]))]
    plt.figure(figsize=(24,12),dpi=600)
    count = 0
    for tab in range(10):
        index = tab
        plt.subplot(1,4,count+1)
        if tab == 9:
            index = -1
        elif tab == 8:
            index = -2
        plt.plot(X,weight_list_new[0][index])
        plt.xlabel('Epoch\n('+subtitle[count]+') Scale '+str(tab+1), fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.grid()
        count += 1

    """
    plt.subplot(2,5,2)
    plt.plot(X,weight_list_new[0][1])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    plt.subplot(2,5,3)
    plt.plot(X,weight_list_new[0][2])
    plt.xlabel('Epoch\n('+subtitle[1]+')', fontsize=10)
    plt.ylabel('Scale Weight', fontsize=10)
    plt.grid()

    """
    plt.tight_layout()
    if corss_val_label == 0:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Weight_list_" + filename + "_SW_" + str(sequence_window) +"_LR_"+str(learning_rate) + ".pdf"), dpi=600)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Weight_list_" + filename + "_SW_" + str(sequence_window) +"_LR_"+str(learning_rate) + ".png"), dpi=600)
    else:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Weight_list_"+filename+"_SW_"+str(sequence_window)+".pdf"),dpi=600)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Weight_list_"+filename+"_SW_"+str(sequence_window)+".png"),dpi=600)

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

def Quxian_Plotting(dataX,dataY,feature,name):
    y = list(dataX[0][:,feature])
    for i in range(1,len(dataX)):
        y.append(dataX[i][:,feature][-1])
    x = [i for i in range(len(y))]
    z = [i  for i in range(len(dataY)) if int(dataY[i][0]) == 1]
    print("1111111111111111111")
    plt.plot(x,np.array(y),'b')
    print("2222222222222222222")
    plt.plot(z,np.array(y)[z],'r')
    print("3333333333333333333")
    plt.tight_layout()
    print("4444444444444444444")
    plt.grid()
    print("5555555555555555555")
    plt.clf()
    #plt.show()
    print("6666666666666666666")
    plt.savefig(name + '.pdf', dpi=400)


#A1 = [51.5,54.2,55.1,55.4,55.8,57.3,49.6,52.2,63.4,63.5]#"AS_LEAK"
#A2 = [50.6,53.9,55.3,54.3,56.8,52.7,54.7,52.1,63.3,63.3]
#A3 = [50.4,54.5,55.7,54.3,56.1,51.0,54.6,51.9,63.3,63.3]
#A = []
#A.append(A1)
#A.append(A2)
#A.append(A3)
#MC_Plotting(A,1,3)
#------------------------------------------Plotting AUC-----------------------------------------------------
#filename_list = ["HB_AS_Leak.txt","HB_Slammer.txt","HB_Nimda.txt","HB_Code_Red_I.txt"]
#method_list = ["SVM","NB","DT","Ada.Boost","MLP","RNN","LSTM","MS-LSTM"]

def plotAUC(method_list,Parameters):
    fpr = [0 for i in range(len(method_list))]
    tpr = [0 for i in range(len(method_list))]
    auc = [0 for i in range(len(method_list))]
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
    #plt.savefig("_AUC.png", dpi=800)
    plt.savefig("_AUC.pdf", dpi=800)
#------------------------------------------Plotting STAT-----------------------------------------------------
def _plotting(filename, subtitle, method):
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
def plotStat(filename, Method_List):
    subtitle = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    plt.figure(figsize=(12, 6), dpi=800)
    plt.subplot(3, 3, 1)
    filename_ = "StatFalseAlarm_" + filename + "_True.txt"
    _plotting(filename_, subtitle[0], "Original")

    for tab in range(len(Method_List)):
        filename_ = "StatFalseAlarm_" + filename + "_" + Method_List[tab] + "_" + "_Predict.txt"
        plt.subplot(3, 3, tab + 1)
        _plotting(filename_, subtitle[tab + 1], Method_List[tab])

    plt.tight_layout()
    plt.savefig("StateFalseAlarm_" + filename + ".pdf", dpi=400)
    #plt.show()
#------------------------------------------Plotting Wavelet-----------------------------------------------------
def PlotWavelet(filename_result,filename_result2):
    filename_list_label = ["AS_Leak","Slammer","Nimda","Code_Red_I"]
    #wave_type_db=[81.66,82.79,85.47,80.25,80.90,81.55,82.80,87.57,92.82,93.35,95.64]
    #wave_type_haar=[81.66,82.79,85.47,80.25,80.90,81.55,82.80,87.57,92.82,93.35,95.64]
    color_type = ['y.', 'gs','#FF8C00','#FD8CD0','c>', 'b<', 'r.', 'm*']
    #wave_type_sym= []
    with open("aaa.txt","a")as fout:
        for eachk,eachv in filename_result.items():
            fout.write(eachk)
            fout.write(''.join(eachv))
        for eachk,eachv in filename_result2.items():
            fout.write(eachk)
            fout.write(''.join(eachv))

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
    plt.ylim(0,100)
    plt.tick_params(labelsize=12)
    plt.xlabel("Scale Levels",fontsize=12)
    plt.grid()
    plt.savefig("Wave_Let_"+'_'+str(0)+'_'+".df",dpi=400)
    plt.title("Wavelet Family: Daubechies/Filter Length: 2")
    #plt.show()




