from __future__ import print_function
import numpy as np
import sys
import os
import cntk
import math
import read_data as rd

abs_path  = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, "data")

#parameters
my_strides = (2, 1)
my_conv_rf_shape = (3, 1)
my_pool_rf_shape = (2, 1)
my_drop = 0.1
learning_rate = 0.001
input_height = 40
input_width  = 1
num_channels = 1
input_dim = input_height * input_width * num_channels
num_output_classes = 4

def net():
    input_var = cntk.ops.input_variable((num_channels, input_height, input_width), np.float32)
    label_var = cntk.ops.input_variable(num_output_classes, np.float32)

    with cntk.layers.default_options(init = cntk.initializer.he_normal(), activation = cntk.ops.relu, bias = True, init_bias = cntk.initializer.he_normal()):
        conv1 = cntk.layers.Convolution(my_conv_rf_shape, num_filters = 64, strides = my_strides, pad = True)(input_var)
        pool1 = cntk.layers.MaxPooling(my_pool_rf_shape, strides = my_strides, pad = True)(conv1)

        conv2 = cntk.layers.Convolution(my_conv_rf_shape, num_filters = 72, strides = my_strides, pad = True, bias = True)(pool1)
        pool2 = cntk.layers.MaxPooling(my_pool_rf_shape, strides = my_strides, pad = True)(conv2)

        conv3 = cntk.layers.Convolution(my_conv_rf_shape, num_filters = 96, strides = my_strides, pad = True)(pool2)
        pool3 = cntk.layers.MaxPooling(my_pool_rf_shape, strides = my_strides, pad = True)(conv3)

        fc1   = cntk.layers.Dense(96)(input_var)
        drop1 = cntk.layers.Dropout(my_drop)(fc1)

        fc2   = cntk.layers.Dense(72)(drop1)
        drop2 = cntk.layers.Dropout(my_drop)(fc2)

        fc3   = cntk.layers.Dense(64)(drop2)
        drop3 = cntk.layers.Dropout(my_drop)(fc3)

        out   = cntk.layers.Dense(num_output_classes, activation = None)(drop3)

    ce = cntk.ops.cross_entropy_with_softmax(out, label_var)
    pe = cntk.ops.classification_error(out, label_var)

    epoch_size = 30000
    minibatch_size = 21

    lr_per_sample    = [learning_rate]
    lr_schedule      = cntk.learning_rate_schedule(lr_per_sample, cntk.learner.UnitType.sample, epoch_size)
    mm_time_constant = [1100]
    mm_schedule      = cntk.learner.momentum_as_time_constant_schedule(mm_time_constant, epoch_size)

    learner = cntk.learner.adam_sgd(out.parameters, lr = lr_schedule, momentum = mm_schedule)
    trainer = cntk.Trainer(out, ce, pe, learner)

    rd.read_data_()
    val_data, val_label = rd.get_val()

    step = 1
    while step * minibatch_size < epoch_size:
        features, labels = rd.next_train_batch(minibatch_size, step - 1)
        trainer.train_minibatch({input_var: features, label_var: labels})
        step += 1
        if step % 30 == 0:
            print(step, "Training Accuracy:", 1 - trainer.test_minibatch({input_var: features, label_var: labels}))
            print(step, "Testing Accuracy:", 1 - trainer.test_minibatch({input_var: val_data, label_var: val_label}))

    print("Final Testing Accuracy:", 1 - trainer.test_minibatch({input_var: val_data, label_var: val_label}))

if __name__ == '__main__':
    net()
