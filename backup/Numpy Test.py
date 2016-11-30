import numpy as np
def batch_vm(v, m):
  shape = tf.shape(v)
  rank = shape.get_shape()[0].value
  v = tf.expand_dims(v, rank)

  vm = tf.mul(v, m)

  return tf.reduce_sum(vm, rank-1)
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
A = np.ones((3,4,5))
#b = np.array([0.1,0.2,0.3,0.4,0.5])


import tensorflow as tf
# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

#x = tf.Variable(np.ones((5,3,4)),dtype='float32')
b = tf.Variable(tf.random_normal(shape=[5]))
b = tf.assign(b,[0.2,0.2,0.2,0.2,0.2])
temp_ = tf.constant(1.0,shape=[5])
bbb = tf.mul(b,temp_)
e = tf.exp(b)
g = tf.gather(tf.cumsum(tf.exp(b)),5-1)

f = tf.mul(temp_, b)
b = tf.assign(b,[0.6,0.1,0.1,0.1,0.1])

scale_weight = tf.div(tf.exp(tf.mul(temp_, b)),
                      tf.gather(tf.cumsum(tf.exp(tf.mul(temp_, b))), 5 - 1))

h = tf.div(tf.exp(b),tf.gather(tf.cumsum(tf.exp(b)),5-1))
#print(x)
#b = tf.reshape(b,(1,5))
# Initialize 'x' using the run() method of its initializer op.
#x.initializer.run()
#b.initializer.run()
#bb = sess.run(re)
print(e.eval())
print(g.eval())
print(h.eval())
print(bbb.eval())
print(f.eval())
print(scale_weight.eval())


#b = tf.reshape(b,(1,5))
#b2 = tf.expand_dims(b,2)
#x2 = tf.reshape(x,(5,3*4))
# Add an op to subtract 'a' from 'x'.  Run it and print the result
#sub = tf.mul(b,x2)
#sub = batch_vm2(b,x2)
#sub2 = tf.reshape(sub,(3,4))
#x3 =tf.reshape(sub,(5,3,4))
#print(x.eval())
#print(b.eval())
#print(x2.eval())
#print(sub.eval())
#print(sub2.eval())
#print(x3.eval())
#print(re.eval().shape)
# ==> [-2. -1.]

# Close the Session when we're done.
sess.close()