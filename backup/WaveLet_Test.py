import pywt
import numpy as np
import matplotlib.pyplot as plt
A = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[10,20,30,40,50,60,70,80],[10,20,30,40,50,60,70,80],[10,20,30,40,50,60,70,80]]
A = np.array(A)
#A = np.transpose(A)
print(A.shape)

def Plottint(Data_A):
    m,n = Data_A.shape
    if min(m,n) <=8:
        number = 2
    else:
        number = 3
    coeffs = pywt.wavedec(Data_A, 'db1', level=number)

    X = [i+1 for i in range(len(Data_A[0]))]
    Selected_Feature = 5
    if number == 2:
        #print(coeffs)
        plt.subplot(2,2,1)
        plt.plot(X,Data_A[Selected_Feature],'b')
        CA0 = coeffs[0]

        R0 = pywt.idwt(pywt.idwt(None,coeffs[1],'db1'),None,'db1')
        print("R0......")
        print(R0)
        plt.subplot(2,2,2)
        plt.plot(X,R0[Selected_Feature],'b')
        CD1 = coeffs[1]
        R1 = pywt.idwt(pywt.idwt(CD1,None,'db1'),None,'db1')
        print("R1......")
        #print(R1)
        plt.subplot(2,2,3)
        plt.plot(X,R1[Selected_Feature],'b')


        CD2 = coeffs[2]
        #R2 = pywt.idwt(None,CD2,'db1')
        #coeffs[2] = None
        #coeffs[0] = None
        coeffs[2] = None

        print(coeffs)
        R2 = pywt.waverec(coeffs,'db1')
        #R2 = pywt.idwt(pywt.idwt(CD2,None,'db1'),None,'db1')

        print("R2......")
        #print(R2)
        plt.subplot(2,2,4)
        plt.plot(X,R2[Selected_Feature],'b')
        plt.show()
        #print(R1+R2)


Plottint(A)
