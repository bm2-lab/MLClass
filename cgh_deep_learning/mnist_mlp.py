import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot=True)
mnist_train = mnist.train
mnist_val = mnist.validation

p = 28 * 28
n = 10
h1 = 300

func_act = tf.nn.sigmoid

x_pl = tf.placeholder(dtype=tf.float32, shape=[None, p])
y_pl = tf.placeholder(dtype=tf.float32, shape=[None, n])

w1 = tf.Variable(tf.truncated_normal(shape=[p, h1], stddev=0.1))
b1 = tf.Variable(tf.zeros(shape=[h1]))

w2 = tf.Variable(tf.truncated_normal(shape=[h1, n], stddev=0.1))
b2 = tf.Variable(tf.zeros(shape=[n]))

hidden1 = func_act(tf.matmul(x_pl, w1) + b1)
y_pre = tf.matmul(hidden1, w2) + b2
y_ = tf.nn.softmax(y_pre)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_pl, logits=y_pre))
correct_prediction = tf.equal(tf.argmax(y_pl, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

eta = 0.3
train_op = tf.train.AdagradOptimizer(learning_rate=0.3).minimize(cross_entropy)

batch_size = 50
batch_per_epoch = mnist_train.num_examples // batch_size
epoch = 2
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    x_val = mnist_val.images
    y_val = mnist_val.labels
    val_fd = {x_pl: x_val, y_pl: y_val}
    for ep in range(epoch):
        print(f'Epoch {ep+1}:')
        for sp in range(batch_per_epoch):
            xtr, ytr = mnist_train.next_batch(batch_size)
            loss_value, _ = sess.run([cross_entropy, train_op], feed_dict={x_pl: xtr, y_pl: ytr})
            if sp == 0 or (sp + 1) % 100 == 0:
                print(f'Loss: {loss_value:.4f}')
        acc = sess.run(accuracy, feed_dict=val_fd)
        print(f'Validation Acc: {acc:.4f}')













