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



