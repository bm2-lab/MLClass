import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from util.plot import show_mnist_fig


mnist = input_data.read_data_sets('data', one_hot=True)
mnist_train = mnist.train
mnist_val = mnist.validation
mnist_test = mnist.test

xte, yte = mnist_test.next_batch(50)

xe = xte[2]
ye = yte[2]
show_mnist_fig('t2', xe, ye)