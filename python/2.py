import numpy as np
import tensorflow as tf
x_data=np.random.rand(100).astype(np.float32) 
y_data=x_data*0.1+0.3
weight=tf.Variable(tf.random_uniform([1],-1.0,1.0),name="w")
biases=tf.Variable(tf.zeros([1]),name="b")
  
y=weight*x_data+biases
  
loss=tf.reduce_mean(tf.square(y-y_data)) #loss
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
  
  
init=tf.global_variables_initializer() 
sess=tf.Session()
sess.run(init)
saver=tf.train.Saver(max_to_keep=0)
for step in range(10):
  sess.run(train)
  saver.save(sess,"./save_mode",global_step=step) #保存
  print("当前进行：",step)