import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('data', one_hot=True)
mnist_train = mnist.train
mnist_val = mnist.validation
# mnist_test = mnist.test

x_pl = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_pl = tf.placeholder(dtype=tf.float32, shape=[None, 10])

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.1), name='weight')
b = tf.Variable(tf.zeros(shape=[10]), name='bias')
y_pre = tf.matmul(x_pl, w) + b
y_hat = tf.nn.softmax(y_pre)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_pl, logits=y_pre)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy_mean)


xval, yval = mnist_val.next_batch(10)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

steps = 100
for i in range(steps):
    xtr, ytr = mnist_train.next_batch(50)
    train_fd = {x_pl:xtr, y_pl:ytr}
    loss, _ = sess.run([cross_entropy_mean, train_op], feed_dict=train_fd)
    if (i+1) % 10 == 0:
        print(f'Loss: {loss}')

















