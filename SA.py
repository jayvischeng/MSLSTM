import os
import matplotlib.pyplot as plt
import matplotlib
def set_style():
    plt.style.use(['classic'])
    matplotlib.rc("font", family="serif")
set_style()
def Plot(filename,subtitle,method):
    #Predict = []
    #True = []
    Temp = []
    with open(os.path.join(os.getcwd(),filename))as fin:
        for each in fin.readlines():
            Temp.append(int(each))
            #val = each.split('\t\t')
            #Predict.append(int(val[0].strip()))
            #True.append(int(float(val[-1].strip())))

    Temp = Temp[2340:4740]

    if "SVM" in filename:
        print(Temp)

    X = [i+1 for i in range(len(Temp))]

    plt.xlim(0,len(X))
    plt.ylim(-0.5,2.0)
    if not method=="Original":
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
        plt.plot(X,Temp,'b.',markersize=2,label='Predict')
    else:
        plt.plot(X, Temp, 'r.', markersize=2,label='True')

    plt.legend(loc=1, fontsize = 12)
    #plt.legend(bbox_to_anchor=(1, 1),
               #bbox_transform=plt.gcf().transFigure)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('('+subtitle+')'+"  "+method)
    x = len(X) / 2
    plt.xticks([1,400,800,1200,1600,2000,2500])
    #plt.grid()
    #plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.1)
    #plt.minorticks_on()
    plt.axvline(x, ymin=-1, ymax=2, linewidth=2, color='g')    #plt.title('Testing Sequence')
    #plt.axvline(x+30, ymin=-1, ymax=2, linewidth=2, color='g')    #plt.title('Testing Sequence')

    #plt.grid()
filename_="HB_Nimda.txt"
Method_List = ["SVM","NB","DT","Ada.Boost","MLP","RNN","LSTM","MS-LSTM"]
filename = "StatFalseAlarm_"+filename_+"_True.txt"
subtitle = ['a','b','c','d','e','f','g','h','i']

plt.figure(figsize=(12,6),dpi=800)
plt.subplot(3,3,1)
Plot(filename,subtitle[0],"Original")

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[0]+"_"+"_Predict.txt"
plt.subplot(3,3,2)
Plot(filename,subtitle[1],Method_List[0])

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[1]+"_"+"_Predict.txt"
plt.subplot(3,3,3)
Plot(filename,subtitle[2],Method_List[1])

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[2]+"_"+"_Predict.txt"
plt.subplot(3,3,4)
Plot(filename,subtitle[3],Method_List[2])

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[3]+"_"+"_Predict.txt"
plt.subplot(3,3,5)
Plot(filename,subtitle[4],Method_List[3])

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[4]+"_"+"_Predict.txt"
plt.subplot(3,3,6)
Plot(filename,subtitle[5],Method_List[4])

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[5]+"_"+"_Predict.txt"
plt.subplot(3,3,7)
Plot(filename,subtitle[6],Method_List[5])

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[6]+"_"+"_Predict.txt"
plt.subplot(3,3,8)
Plot(filename,subtitle[7],Method_List[6])

filename = "StatFalseAlarm_"+filename_+"_"+Method_List[7]+"_"+"_Predict.txt"
plt.subplot(3,3,9)
Plot(filename,subtitle[8],Method_List[7])

plt.tight_layout()
plt.savefig("StateFalseAlarm_"+filename+".png",dpi=400)
plt.show()


