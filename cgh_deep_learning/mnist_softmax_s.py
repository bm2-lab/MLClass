import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot=True)
mnist_train = mnist.train
mnist_val = mnist.validation
mnist_test = mnist.test

log_dir = 'temp/softmax'
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
else:
    tf.gfile.MakeDirs(log_dir)


p = 28 * 28
n = 10

x_pl = tf.placeholder(dtype=tf.float32, shape=[None, p])
y_pl = tf.placeholder(dtype=tf.float32, shape=[None, n])

w = tf.Variable(tf.random_normal(shape=[p, n]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[n]), name='bias')
y_pre = tf.matmul(x_pl, w) + b
y_ = tf.nn.softmax(y_pre)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_pl, logits=y_pre))
correct_prediction = tf.equal(tf.argmax(y_pl, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Loss', cross_entropy)

eta = 0.1
train_op = tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(cross_entropy)

batch_size = 50
batch_per_epoch = mnist_train.num_examples // batch_size
epoch = 10

summary = tf.summary.merge_all()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    tf.global_variables_initializer().run()
    x_val = mnist_val.images
    y_val = mnist_val.labels
    val_fd = {x_pl: x_val, y_pl: y_val}
    for ep in range(epoch):
        print(f'Epoch {ep+1}:')
        for sp in range(batch_per_epoch):
            xtr, ytr = mnist_train.next_batch(batch_size)
            i = sp + ep * batch_per_epoch + 1
            loss_value, summary_value, _ = sess.run([cross_entropy, summary, train_op], feed_dict={x_pl: xtr, y_pl: ytr})
            if sp == 0 or (sp + 1) % 100 == 0:
                train_writer.add_summary(summary_value, i)
                print(f'Loss: {loss_value:.4f}')
        acc = sess.run(accuracy, feed_dict=val_fd)
        print(f'Validation Acc: {acc:.4f}')

train_writer.close()



