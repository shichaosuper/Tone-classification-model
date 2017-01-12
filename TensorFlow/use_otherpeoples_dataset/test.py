'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
import read_data as rd

# Parameters
learning_rate = 0.001
training_iters = 30000
batch_size = 21
display_step = 1
# Network Parameters
n_input = 36
n_classes = 4 
dropout = 0.8

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, 2])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, 1, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 1, n_input, 2])
    
    
    
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    tf.histogram_summary("wc1",weights['wc1'])
    tf.histogram_summary("bc1",biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    tf.histogram_summary("wc2",weights['wc2'])
    tf.histogram_summary("bc2",biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    tf.histogram_summary("wc3",weights['wc3'])
    tf.histogram_summary("bc3",biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
      
      
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    tf.histogram_summary("wd1",weights['wd1'])
    tf.histogram_summary("bd1",biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    tf.histogram_summary("wd2",weights['wd2'])
    tf.histogram_summary("bd2",biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    tf.histogram_summary("wd3",weights['wd3'])
    tf.histogram_summary("bd3",biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out


weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 3, 2, 256],mean=-0.0,stddev=1.0)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([1, 3, 256, 128],mean=0.00,stddev=1.0)),
    'wc3': tf.Variable(tf.random_normal([1, 3, 128, 64],mean=0.00,stddev=1.0)),
    'wd1': tf.Variable(tf.random_normal([1*64, 128],mean=0.00,stddev=1.0)),#37
    'wd2': tf.Variable(tf.random_normal([128, 64],mean=0.0,stddev=1.0)),
    'wd3': tf.Variable(tf.random_normal([64, 64],mean=0.0,stddev=1.0)),
    # 300 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([64, n_classes],mean=0.0,stddev=1.0))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([256],mean=0.0,stddev=1.0)),
    'bc2': tf.Variable(tf.random_normal([128],mean=0.0,stddev=1.0)),
    'bc3': tf.Variable(tf.random_normal([64],mean=0.0,stddev=1.0)),
    'bd1': tf.Variable(tf.random_normal([128],mean=0.0,stddev=1.0)),
    'bd2': tf.Variable(tf.random_normal([64])),
    'bd3': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
tf.scalar_summary('loss',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-02).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
merged = tf.merge_all_summaries()

rd.read_data_()
val_data, val_label = rd.get_val()
# Launch the graph
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("./log/",sess.graph)
    sess.run(init)
    step = 1
    max_ = 0.1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = rd.next_train_batch(batch_size, step - 1)
        print( batch_x.shape, batch_y.shape)
        # Run optimization op (backprop)
        loss, acc_ = sess.run([merged,optimizer], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        writer.add_summary(loss, step)
        writer.add_summary(acc_, step)
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        if(step % 6 == 0):
          num_ = sess.run(accuracy, feed_dict={x: val_data,
                                        y: val_label,
                                        keep_prob: 1.})
          if (num_ > max_):
              max_ = num_
          print("Testing Accuracy:", num_)
    print("Optimization Finished!")
    print("max acc:", max_)
    
    
    # Calculate accuracy for 300 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: val_data,
                                      y: val_label,
                                      keep_prob: 1.}))