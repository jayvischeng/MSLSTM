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