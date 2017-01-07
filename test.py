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
training_iters = 100000
batch_size = 21
display_step = 1
# Network Parameters
n_input = 128
n_classes = 4 
dropout = 0.7

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, 2])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 1, 128, 2])
    fc1 = tf.reshape(x, [-1, 2*128])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out


weights = {

    'wd1': tf.Variable(tf.random_normal([2*128, 512])),#37
    'wd2': tf.Variable(tf.random_normal([512, 128])),
    'wd3': tf.Variable(tf.random_normal([128, 64])),
    # 512 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([64, n_classes]))
}

biases = {
    'bd1': tf.Variable(tf.random_normal([512])),
    'bd2': tf.Variable(tf.random_normal([128])),
    'bd3': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
rd.read_data_()
val_data, val_label = rd.get_val()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = rd.next_train_batch(batch_size, step - 1)
        print( batch_x.shape, batch_y.shape)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
        if(step % 20 == 0):
          print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: val_data,
                                        y: val_label,
                                        keep_prob: 1.}))
    print("Optimization Finished!")
    
    
    

    # Calculate accuracy for 512 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: val_data,
                                      y: val_label,
                                      keep_prob: 1.}))