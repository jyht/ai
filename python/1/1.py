import tensorflow as tf

saver = tf.train.Saver()
saver.save(sess,"./checkpoint_dir/MyModel")
