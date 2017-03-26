import matplotlib
#matplotlib.use('GTKAgg')
#matplotlib.rcParams['backend'] = 'GTKCairo'
import os
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pylab as p
from sklearn import decomposition
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()


def reverse_one_hot(y):
    temp = []
    for i in range(len(y)):
        for j in range(len(y[0])):
            if y[i][j] == 1:
                temp.append(j)
    return np.array(temp)

def epoch_acc_plotting(filename,case_list,sequence_window,learning_rate,train_acc_list,val_acc_list):
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
        plt.ylim(0.3,1.05)
    else:
        plt.ylim(0.05,1.05)
    plt.ylabel('Accuracy',fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.legend(loc='lower right',fontsize=8)
    plt.title(filename.split('.')[0].replace('HB_','')+'/sw: '+str(sequence_window))
    plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Epoch_ACC_"+filename + "_SW_"+str(sequence_window)+".pdf"),dpi=400)

    #if corss_val_label == 0:
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"2Tab_A_Epoch_ACC_"+filename + "_SW_"+str(sequence_window)+".pdf"),dpi=400)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Epoch_ACC_"+filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png"),dpi=400)
    #else:
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"2Tab_B_Epoch_ACC_" + filename + "_SW_"+str(sequence_window)+".pdf"), dpi=400)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Epoch_ACC_" + filename + "_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png"), dpi=400)

def epoch_loss_plotting(filename,case_list,sequence_window,learning_rate,train_loss_list,val_loss_list):
    if not os.path.isdir(os.path.join(os.getcwd(),'picture')):
        os.makedirs(os.path.join(os.getcwd(),'picture'))
    epoch = len(train_loss_list[0])
    color_list = ['y', 'g','#FF8C00','#FD8CD0','c', 'b', 'r', 'm']

    x = [i+1 for i in range(epoch)]
    plt.figure()
    for tab in range(len(case_list)):
        plt.plot(x,train_loss_list[tab],color_list[tab],label=case_list[tab] + ' train loss')
        plt.plot(x, val_loss_list[tab], color_list[len(case_list)+tab],label=case_list[tab] +' val loss')

    plt.xlabel('Epoch',fontsize=12)
    plt.ylabel('Loss',fontsize=12)
    plt.grid()
    plt.tick_params(labelsize=12)
    plt.legend(loc='upper right',fontsize=8)
    plt.title(filename.split('.')[0].replace('HB_','')+'/sw: '+str(sequence_window))
    plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Epoch_Loss_"+filename+"_SW_"+str(sequence_window)+"_LR_"+".pdf"),dpi=400)

    #if cross_val_label == 0:
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"2Tab_A_Epoch_Loss_"+filename+"_SW_"+str(sequence_window)+"_LR_"+".pdf"),dpi=400)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Epoch_Loss_"+filename+"_SW_"+str(sequence_window)+"_LR_"+str(learning_rate)+".png"),dpi=400)
    #else:
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"2Tab_B_Epoch_Loss_" + filename + "_SW_" + str(sequence_window) + ".pdf"), dpi=400)
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
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"ATab_A_Weight_list_" + filename + "_SW_" + str(sequence_window) +"_LR_"+str(learning_rate) + ".pdf"), dpi=600)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_A_Weight_list_" + filename + "_SW_" + str(sequence_window) +"_LR_"+str(learning_rate) + ".png"), dpi=600)
    else:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"ATab_B_Weight_list_"+filename+"_SW_"+str(sequence_window)+".pdf"),dpi=600)
        #plt.savefig(os.path.join(os.path.join(os.getcwd(),'picture'),"Tab_B_Weight_list_"+filename+"_SW_"+str(sequence_window)+".png"),dpi=600)
def curve_plotting_withWindow(dataX,dataY,feature,name):
    y = list(dataX[0][:,feature])
    for i in range(1,len(dataX)):
        y.append(dataX[i][:,feature][-1])
    x = [i for i in range(len(y))]
    z = [i  for i in range(len(dataY)) if int(dataY[i][0]) == 1]
    print(len(y))
    plt.plot(x,np.array(y),'b')
    plt.plot(z,np.array(y)[z],'r.')
    plt.tight_layout()
    plt.grid()
    plt.show()
    plt.savefig(name + '.pdf', dpi=400)
def curve_plotting(dataX,dataY,name,method):
    np.random.seed(5)
    target_names = ['Regular','Anomalous']
    centers = [[1, 1], [-1, -1]]
    X = dataX
    y = dataY
    plt.clf()
    plt.cla()
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    try:
        y = reverse_one_hot(y)
    except:
        pass
    print(y[0])
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=2,
                    label=target_name)
    plt.legend(loc='lower right', fontsize = 16, shadow=False, scatterpoints=1)
    plt.tick_params(labelsize = 15)
    plt.grid()
    #plt.title('PCA of the dataset')
    plt.savefig(name + method +"_PCA.pdf", dpi=400)
    plt.show()
def plotAUC(results,method_list,filename):

    plt.figure()
    #color_list = ['y', 'g', '#FF8C00', 'c', 'b', 'r', 'm']
    color_list = ['y', 'g','#FF8C00','#FD8CD0','c', 'b', 'r', 'm']
    for tab in range(len(method_list)):
        fpr = results[method_list[tab]]['FPR']
        tpr = results[method_list[tab]]['TPR']
        auc = results[method_list[tab]]['AUC']
        plt.plot(fpr, tpr, color_list[tab], label=method_list[tab] + ' ROC curve (area = %0.2f)' % auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    plt.title('Receiver operating characteristic of '+filename.replace('HB_','').split('.')[0],fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.tick_params(labelsize=12)
    plt.grid()
    #plt.savefig("_AUC.png", dpi=800)
    plt.savefig(filename+"TP_FP_AUC.pdf", dpi=800)
#------------------------------------------Plotting STAT-----------------------------------------------------
def _plotting(filename, subtitle, method,method_dict):
    temp = []
    try:
        with open(os.path.join(os.path.join(os.getcwd(),'stat'), filename))as fin:
            for each in fin.readlines():
                temp.append(int(each))
    except:
        with open(os.path.join(os.path.join(os.getcwd(),'stat'), filename).replace('Predict.txt','Predict'))as fin:
            for each in fin.readlines():
                temp.append(int(each))
    temp = np.array(temp)
    X = [i + 1 for i in range(len(temp))]
    X = np.array(X)
    plt.xlim(0, len(X))
    plt.ylim(-0.5, 2.0)
    #if  'True' in method:
        #p1_,= plt.plot(X[temp==0], temp[temp==0], 'b.', markersize=2, label='Regular')
        #l1 = plt.legend([p1_], ["Regular"])
        #p2_,= plt.plot(X[temp==1], temp[temp==1], 'r.', markersize=2,label='Anomalous')
        #plt.legend()
        #plt.gca().add_artist(l1)
        #l2 = plt.legend([p2_], ["Anomalousppppppp"])
        #plt.gca().add_artist(l1)
    if 1>0:
        p1, = plt.plot(X[temp==1], temp[temp==1], 'b.', markersize=4, label='Regular')
        p2, = plt.plot(X[temp==0], temp[temp==0], 'r.', markersize=4, label='Anomalous')
        l1 = plt.legend([p1], ["Regular"], loc=2,fontsize=10)
        plt.gca().add_artist(l1)
        l2 = plt.legend([p2], ["Anomalous"], loc=0,fontsize=10)
        plt.gca().add_artist(l2)

    #plt.legend(loc=1, fontsize=12)
    #plt.legend(bbox_to_anchor=(1, 1),
    #bbox_transform=plt.gcf().transFigure)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    try:
        plt.xlabel('(' + subtitle + ')' + "  " + method_dict[method]+' Predicted',fontsize=10)
    except:
        if 'True' in method:
            plt.xlabel('(' + subtitle + ')' + "  " + method,fontsize=10)
        else:
            plt.xlabel('(' + subtitle + ')' + "  " + method+' Predicted',fontsize=10)

    x = (len(X) / 2)
    #plt.xticks([1, 400, 800, 1200, 1600, 2000, 2500])
    plt.xticks([1, 100, 200, 300, 400, 500, 600,700,800])
    plt.tick_params(labelsize=10)
    # plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.1)
    # plt.minorticks_on()
    #plt.grid()
    plt.grid(b=True, which='minor')
    plt.axvline(x, ymin=-1, ymax=2, linewidth=2, color='g')  # plt.title('Testing Sequence')
    # plt.axvline(x+30, ymin=-1, ymax=2, linewidth=2, color='g')    #plt.title('Testing Sequence')
    # plt.grid()
def plotStat(filename, Method_List,Method_Label):
    subtitle = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    plt.figure(figsize=(15, 4), dpi=400)
    #plt.figure()
    plt.subplot(2, 4, 1)
    filename_ = "StatFalseAlarm_" + filename + "_True.txt"
    _plotting(filename_, subtitle[0], "True Label",Method_Label)

    for tab in range(len(Method_List)):
        filename_ = "StatFalseAlarm_" + filename + "_" + Method_List[tab] + "_" + "_Predict.txt"
        plt.subplot(2, 4, tab + 2)
        _plotting(filename_, subtitle[tab + 1], Method_List[tab],Method_Label)

    plt.tight_layout()
    plt.savefig("StateFalseAlarm_" + filename + ".pdf", dpi=400)
    plt.show()
#------------------------------------------Plotting Wavelet-----------------------------------------------------
def plotWavelet(filename_result,filename_result2):
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

#A1 = [51.5,54.2,55.1,55.4,55.8,57.3,49.6,52.2,63.4,63.5]#"AS_LEAK"
#A2 = [50.6,53.9,55.3,54.3,56.8,52.7,54.7,52.1,63.3,63.3]
#A3 = [50.4,54.5,55.7,54.3,56.1,51.0,54.6,51.9,63.3,63.3]
#A = []
#A.append(A1)
#A.append(A2)
#A.append(A3)
#MC_Plotting(A,1,3)

def plot3D(X=[],Y=[],Z=[]):

    X = [1, 2, 3, 4, 5, 6]
    Y = [10, 20, 30, 40]
    X = np.array(X)
    Y = np.array(Y)
    Z1 = [1.28982, 1.03682, 0.99231, 0.9999, 1.01086, 1.0222, 1.18597, 0.9997, 1.0195, 1.04, 1.11, 1.15133, 1.1123,
          1.04807, 1.04694, 1.0245, 1.01838, 1.01737, 1.18136, 1.05632, 1.01083, 0.99859, 0.99589, 0.99618]
    Z1 = np.array(Z1)

    max_ = np.max(Z1)
    min_ = np.min(Z1)

    XX, YY = np.meshgrid(X, Y)
    Z1 = np.reshape(Z1, (XX.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(XX, YY, Z1, rstride=1, cstride=1, alpha=1, cmap=cm.jet, linewidth=0.5, antialiased=False)
    fig.colorbar(surf, shrink=0.6, aspect=6)
    surf.set_clim(vmin=min_, vmax=max_)
    plt.xlabel('scale level')
    plt.ylabel('window size')
    plt.tight_layout()
    plt.savefig('wqf.pdf', dpi=400)
    plt.show()
#plot3D()



def plotConfusionMatrix(confmat):
    import seaborn
    seaborn.set_context('poster')
    #seaborn.set_style("white")
    seaborn.set_style("ticks")
    plt.style.use(['seaborn-paper'])
    font = {'family': 'serif',
            #'weight': 'bold',
            'size': 12}
    matplotlib.rc("font", **font)

    fig, ax = plt.subplots()
    labels = ['','Regular','AS Leak','  Code Red I','Nimda','Slammer']
    #labels = ['a','b','c','d','e']
    #ticks=np.linspace(0, 5,num=5)
    #res = plt.imshow(confmat, interpolation='none')
    #res = ax.imshow(np.array(confmat), cmap=plt.cm.jet,interpolation='nearest')
    res = ax.imshow(np.array(confmat), interpolation='nearest')

    #plt.xlabel('kkk')
    width, height = confmat.shape

    #plt.xticks(labels)
    #plt.tick_params(labelbottom=labels,labelleft=labels)
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(confmat[x][y]), xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.tick_params(labelsize=10)
    plt.colorbar(res,shrink=1, pad=.01, aspect=10)
    plt.savefig("Fig_10.pdf",dpi=400)
    #plt.show()
    print(confmat.shape)




