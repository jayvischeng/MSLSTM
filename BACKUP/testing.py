import pywt
import sys
import numpy as np
# mul wave test----------------------------------------------------------------------------------------------
def Multi_Scale_Wavelet(X_Training0,Y_Training=[],Level=2,Is_Wave_Let=True):
    print("X_Training")
    NN = X_Training0.shape[0]
    print(X_Training0.shape)
    Level2 = Level + 1
    TEMP_XData = [[] for i in range(Level2)]
    wave_let = 'haar'
    X_Training = X_Training0
    if Is_Wave_Let == True:
        for tab in range(Level2):
            X = []
            for each_feature in range(len(X_Training[0])):
                #print(str(each_feature+1)+" is processing"+" length is "+str(len(X_Training[:,each_feature])))
                coeffs = pywt.wavedec(X_Training[:,each_feature],wave_let, mode='zpd', level=Level)
                current_level = Level2 - tab
                print("-------------------------------------------------+++++++++++++++++++++++" + str(tab))
                print(coeffs)
                for tab2 in range(tab+1,Level2+1):
                    try:
                        pass
                        #coeffs[tab2] = None
                    except:
                        pass

                X_WAVELET_REC = pywt.waverec(coeffs, wave_let)
                print("-------------------------------------------------"+str(tab))
                print(X_WAVELET_REC)
                #print("recoverying legnth is "+str(len(X_WAVELET_REC)))
                X.append(X_WAVELET_REC[:NN])
                #print(np.transpose(np.array(X)).shape)

            TEMP_XData[current_level - 1].extend(np.transpose(np.array(X)))

        TEMP_XData = np.array(TEMP_XData)
        #print("11111111111")
        #print(TEMP_XData.shape)
    else:
        for tab in range(Level):
            current_level = Level - tab
            TEMP_XData[current_level - 1].extend(X_Training)

        TEMP_XData = np.array(TEMP_XData)
    #print("22222222222222")
    #print(TEMP_XData)
    #print("33333333333333")
    return  TEMP_XData,X_Training,Y_Training
def waveFn(wavelet):
    if not isinstance(wavelet, pywt.Wavelet):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet

# given a single dimensional array ... returns the coefficients.
def wavedec(data, wavelet, mode='sym'):
    wavelet = waveFn(wavelet)

    dLen = len(data)
    coeffs = np.zeros_like(data)
    #print("max  level is ")
    level = pywt.dwt_max_level(dLen, wavelet.dec_len)
    #print("max  level is "+str(level))

    a = data
    end_idx = dLen
    for idx in xrange(level):
        a, d = pywt.dwt(a, wavelet, mode)
        begin_idx = end_idx/2
        coeffs[begin_idx:end_idx] = d
        end_idx = begin_idx

    coeffs[:end_idx] = a
    return coeffs

def waverec(data, wavelet, mode='sym'):
    wavelet = waveFn(wavelet)

    dLen = len(data)
    level = pywt.dwt_max_level(dLen, wavelet.dec_len)

    end_idx = 1
    a = data[:end_idx] # approximation ... also the original BGP_Data
    d = data[end_idx:end_idx*2]
    for idx in xrange(level):
        a = pywt.idwt(a, d, wavelet, mode)
        end_idx *= 2
        d = data[end_idx:end_idx*2]
    return a

def fn_dec(arr):
    print(arr)
    coeffs = (pywt.wavedec(arr, 'haar', 'zpd'))
    coeffs[3] = None
    coeffs[2] = None

    bbb = pywt.waverec(coeffs,'haar')
    print(pywt.wavedec(arr, 'haar', 'zpd'))
    print(bbb)
    print(pywt.wavedec(bbb, 'db1', 'zpd',1))
    print(pywt.waverec(coeffs,'haar'))


    return np.array(map(lambda row: reduce(lambda x,y : np.hstack((x,y)), pywt.wavedec(row, 'haar', 'zpd')), arr))
    # return np.array(map(lambda row: row*2, arr))

if __name__ == '__main__':
    test  = -1
    np.random.seed(10)
    wavelet = waveFn('haar')
    if test==0:
        # SIngle dimensional test.
        a = np.random.randn(1,8)
        print "original values A"
        print a[0]
        print "decomposition of A by method in pywt"
        print fn_dec(a[0])
        print " decomposition of A by my method"
        coeffs =  wavedec(a[0], 'haar', 'zpd')
        print coeffs
        print "recomposition of A by my method"
        print waverec(coeffs, 'haar', 'zpd')
        sys.exit()
    if test==1:
        a = np.random.randn(16,16)
        # 2 D test
        print "original value of A"
        print a

        # decompose the signal into wavelet coefficients.
        dimensions = a.shape
        for dim in dimensions:
            a = np.rollaxis(a, 0, a.ndim)
            ndim = a.shape
            #a = fn_dec(a.reshape(-1, dim))
            print("aaa")
            print(a.reshape(-1, dim))
            a = np.array(map(lambda row: wavedec(row, wavelet), a.reshape(-1, dim)))
            a = a.reshape(ndim)
            print("bbb")
            print(a)
        print " decomposition of signal into coefficients"
        print a

        # re-composition of the coefficients into original signal
        for dim in dimensions:
            a = np.rollaxis(a, 0, a.ndim)
            ndim = a.shape
            a = np.array(map(lambda row: waverec(row, wavelet), a.reshape(-1, dim)))
            a = a.reshape(ndim)
        print "recomposition of coefficients to signal"
        print a

#print(Multi_Scale_Wavelet(a))

#A = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
A = [80,61,75,71,63,59,76,63]
a = pywt.wavedec(A,'haar',level=1)
b = pywt.dwt(A,'haar')
#a[1] = None
#A1 = pywt.waverec(a,'db1')
#a = pywt.wavedec(A,'db1',level=1)
#a[0] = None
#D1 = pywt.waverec(a,'db1')

print(a)
print(b)
#print(D1)
#a = pywt.wavedec(A,'db1',level=1)

#print(a[0])
#print(a[1])

#A0 = pywt.idwt(a[0],a[1],'db1')
#A0 = pywt.downcoef(A,a,'db1')
#print(A0)

#A0 =

# numpy test----------------------------------------------------------------------------------------------
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
print(plt.style.available)
def set_style():
    plt.style.use(['classic'])
    matplotlib.rc("font", family="monospace")
set_style()

#matplotlib.rcParams['backend'] = 'GTKCairo'
"""
#A = np.array([[1,2,1],[2,0,0]])
#B = np.array([[[1,2,1],[2200,10,1]],
             # [[123,22,1],[56,33,2]]])
#B = np.ones(shape=(1,2))
#print(B[0])
#print(np.mean(A,axis = 0))
#print(B.shape)
#C = np.sum(B,axis = 0)
#print(np.divide(C,2))
#print(np.divide(np.matmul(B,A),2)[0])

A = np.array([-0.72879553, 0.76005036, -0.0055208355, -0.94793725, 0.77469784, -0.0091524664])
W = np.array([0.58417487, -0.843483, 1.0772164, -1.7771397, 0.86830336, -0.23972222])
b = np.array([0.15928687, 0.15937421, 0.040713146, 0.040625796])
#[0.5474878 -0.5752914 0.3135221 -0.53435749]
u_w=np.array([-1.069649, -0.19398327])
u= np.array([0.43971452,0.56028545])
#[1.4156754]
#m_total:[-0.8450042 0.77619177 -0.0041841217 -0.85157746 0.76825714 -0.0075555854]
val= np.array([-0.72900885, 0.7607016, -0.0030333481, -0.936038, 0.78834856, -0.0050872546, -0.72879553, 0.76005036, -0.0055208355, -0.94793725, 0.77469784, -0.0091524664])
val = np.reshape(val,(6,2))
A = np.reshape(A,(2,3))
W = np.reshape(W,(3,2))
b = np.reshape(b,(2,2))

#print(A*W.T)
T = (A.dot(W)+b)
print(T)
print(u_w.T)
print(T.dot(u_w.T))
#[-0.47402287 -0.23170222]
print(np.exp(-0.47402287)+np.exp(-0.23170222))
print(np.exp(-0.47402287)/1.41567529795)
print((u.T).dot((val.T)))
print(val)
"""
#AS_Leak
wave_type = ["db1","haar","sym","coif","bior","rbio"]
wave_type_db=[81.66,82.79,85.47,80.25,80.90,81.55,82.80,87.57,92.82,93.35,95.64]
wave_type_haar=[81.66,82.79,85.47,80.25,80.90,81.55,82.80,87.57,92.82,93.35,95.64]

wave_type_haar= []
X = [i+2 for i in range(len(wave_type_db))]
plt.plot(X,wave_type_db,'b',label='db1')
#plt.plot(X,wave_type_haar,label='haar')
plt.legend(loc='upper left')
plt.ylabel("Accuracy")
plt.ylim(50,100)
plt.xlabel("multi-scale levels")
plt.grid()
plt.savefig("AAA.png",dpi=800)
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
# tensorflow test----------------------------------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

def batch_vm2(x, m):
  [input_size, output_size] = m.get_shape().as_list()

  input_shape = tf.shape(x)
  batch_rank = input_shape.get_shape()[0].value - 1
  batch_shape = input_shape[:batch_rank]
  output_shape = tf.concat(0, [batch_shape, [output_size]])

  x = tf.reshape(x, [-1, input_size])
  y = tf.matmul(x, m)

  y = tf.reshape(y, output_shape)

  return y




A = tf.Variable(tf.truncated_normal([2,3,4,5]))
#B = tf.tranpose(tf.truncated_normal([2,3,4]))

#B = tf.Variable(tf.constant(0.1,shape = [1,10]))
#A = tf.transpose(A,[1,0])
#C = batch_vm2(A,tf.transpose(B))
#D = [tf.gather(B,i) for i in range(1)]
#C = tf.reshape(C,(1000,30,33))
#C = B.get_shape()[1]-1
#D = tf.gather(tf.gather(tf.cumsum(B,axis = 1),0),0)
#C = tf.div(B,tf.gather(tf.gather(tf.cumsum(B,axis = 1),0),9))



#B = tf.gather(tf.transpose(A,[1,2,0]),[0])
B = tf.gather(tf.reshape(A,(6,4,5)),[0])



output_A = tf.Print(A,[A],message = "A shape is :", first_n=4096, summarize=40)
output_B = tf.Print(B,[B],message = "B  is :", first_n=4096, summarize=40)
#output_C = tf.Print(C,[C],message = "C shape is :", first_n=4096, summarize=40)
#output_D = tf.Print(D,[D],message = "D shape is :", first_n=4096, summarize=40)

#B = C
#C = tf.div(B,tf.gather(tf.gather(tf.cumsum(B,axis = 1),0),5))
#output_B = tf.Print(B,[B],message = "B shape is :", first_n=4096, summarize=40)
#output_C = tf.Print(C,[C],message = "C shape is :", first_n=4096, summarize=40)


sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(output_A)
sess.run(output_B)
#sess.run(output_D)



"""

a = tf.Variable(tf.constant(0.1,shape=[2,5,256]))
b = tf.Variable(tf.constant(0.1,shape=[10]))
output_b = tf.Print(b, [b], message="This is Scale_Weight: ", first_n=1024,summarize=10)

d = tf.transpose(a, [1, 0, 2])
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(b.get_shape())
sess.run(output_b)
x = float(33)
sess.run(tf.scatter_update(b,tf.constant([1]),tf.constant([x])))
sess.run(output_b)
"""
#c = tf.matmul(a,b)
# tf.reshape(tf.matmul(tf.reshape(Aijk,[i*j,k]),Bkl),[i,j,l])

#c = batch_vm2(a,tf.transpose(b))
#c = tf.reshape(tf.matmul(tf.reshape(a,[2*5,256]),b),[2,5,5])
#run = tf.gather(a,1)
#sess = tf.Session()
#init_op = tf.initialize_all_variables()
#sess.run(init_op)
#print(tf.transpose(b).get_shape())
#print(c.get_shape())
"""
def set_style():
    plt.style.use(['seaborn-paper'])
    matplotlib.rc("font", family="serif")
set_style()
A1 = [51.5,54.2,55.1,55.4,55.8,57.3,49.6,52.2,63.4,63.5]#"AS_LEAK"
A2 = [50.6,53.9,55.3,54.3,56.8,52.7,54.7,52.1,63.3,63.3]
A3 = [50.4,54.5,55.7,54.3,56.1,51.0,54.6,51.9,63.3,63.3]
B1 = [71.4,74.3,75.4,79.1,81.8,82.2,88.9,64.7,50.1,85.3]#"Code Red I"
B2 = [72.9,76.2,78.1,80.7,82.6,87.1,82.3,58.7,50.3,85.3]
B3 = [72.2,75.5,77.8,79.9,83.4,87.2,82.8,59.5,50.5,85.2]
#C1 = [58.7,59.4,59.6,61,60.8,60.1,63.8,65.5,50,64.5,60.4,73.7]#"Nimda"
#C2 = [58.4,59.3,59.4,61.5,60.2,59.2,60.6,58.7,57.1,64.6,60.5,73.6]
#C3 = [58.6,59.2,59.8,61.6,60.2,59.9,61.2]
D1 = [66.0,65.9,65.6,65.7,65.3,66.4,60.3,43.4,60.2,60.3]#"Slammer"
D2 = [63.8,65.2,65.7,65.3,63.4,68.7,72.2,66.3,60.1,60.3]
D3 = [64.9,65.6,65.2,65.5,64.0,68.6,72.3,56.0,85.9,60.2]

X1 = [i+1 for i in range(len(A1))]
X2 = [i+1 for i in range(len(B1))]
X3 = [i+1 for i in range(len(D1))]
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(X2,B1,'s-')
plt.xlabel('scale\n(a) window 10',fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.ylim(40,100)
plt.grid()
plt.tick_params(labelsize=10)
plt.subplot(132)
plt.plot(X2,B2,'s-')
plt.xlabel('scale\n(b) window 20',fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.ylim(40,100)
plt.grid()
plt.tick_params(labelsize=10)
plt.subplot(133)
plt.plot(X2,B3,'s-')
plt.xlabel('scale\n(c) window 30',fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.ylim(40,100)
plt.grid()
plt.tick_params(labelsize=10)
plt.tight_layout()
plt.suptitle("AS Leak")
plt.savefig("BBB.png",dpi=200)
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(X3,D1,'s-')
plt.xlabel('scale\n(a) window 10',fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.ylim(40,100)
plt.grid()
plt.tick_params(labelsize=10)
plt.subplot(132)
plt.plot(X3,D2,'s-')
plt.xlabel('scale\n(b) window 20',fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.ylim(40,100)
plt.grid()
plt.tick_params(labelsize=10)
plt.subplot(133)
plt.plot(X3,D3,'s-')
plt.xlabel('scale\n(c) window 30',fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.ylim(40,100)
plt.grid()
plt.tick_params(labelsize=10)
plt.tight_layout()
plt.suptitle("Slammer")
plt.savefig("DDD.png",dpi=200)
plt.show()
"""



