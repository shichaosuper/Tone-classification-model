from __future__ import print_function
import numpy as np
import sys
import os
import cntk
import read_data as rd

abs_path  = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, "data")

#parameters
my_strides = (2, 1)
my_rf_shape = (3, 1)
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

    with cntk.layers.default_options():
        conv1 = cntk.layers.Convolution(my_rf_shape, activation = cntk.ops.relu, num_filters = 64, strides = my_strides, pad = True, bias = True)(input_var)
        sys.stderr.write("conv1" + str(conv1.shape) + '\n')
        pool1 = cntk.layers.MaxPooling(my_rf_shape, strides = my_strides, pad = True)(conv1)
        sys.stderr.write("pool1" + str(pool1.shape) + '\n')
        conv2 = cntk.layers.Convolution(my_rf_shape, activation = cntk.ops.relu, num_filters = 72, strides = my_strides, pad = True, bias = True)(pool1)
        sys.stderr.write("conv2" + str(conv2.shape) + '\n')
        pool2 = cntk.layers.MaxPooling(my_rf_shape, strides = my_strides, pad = True)(conv2)
        sys.stderr.write("pool2" + str(pool2.shape) + '\n')
        conv3 = cntk.layers.Convolution(my_rf_shape, activation = cntk.ops.relu, num_filters = 96, strides = my_strides, pad = True, bias = True)(pool2)
        sys.stderr.write("conv3" + str(conv3.shape) + '\n')
        pool3 = cntk.layers.MaxPooling((2, 1), strides = my_strides, pad = True)(conv3)
        sys.stderr.write("pool3" + str(pool3.shape) + '\n')
        fc1   = cntk.layers.Dense(96, activation = cntk.ops.relu, bias = True)(pool2)
        sys.stderr.write("fc1" + str(fc1.shape) + '\n')
        drop1 = cntk.layers.Dropout(my_drop)(fc1)
        sys.stderr.write("drop1" + str(drop1.shape) + '\n')
        fc2   = cntk.layers.Dense(72, activation = cntk.ops.relu, bias = True)(drop1)
        sys.stderr.write("fc2" + str(fc2.shape) + '\n')
        drop2 = cntk.layers.Dropout(my_drop)(fc2)
        sys.stderr.write("drop2" + str(drop2.shape) + '\n')
        fc3   = cntk.layers.Dense(64, activation = cntk.ops.relu, bias = True)(drop2)
        sys.stderr.write("fc3" + str(fc3.shape) + '\n')
        drop3 = cntk.layers.Dropout(my_drop)(fc3)
        sys.stderr.write("drop3" + str(drop3.shape) + '\n')
        out   = cntk.layers.Dense(num_output_classes)(drop3)

    ce = cntk.ops.cross_entropy_with_softmax(out, label_var)
    pe = cntk.ops.classification_error(out, label_var)

    epoch_size = 30000
    minibatch_size = 21

    lr_per_sample    = [learning_rate]
    lr_schedule      = cntk.learning_rate_schedule(lr_per_sample, cntk.learner.UnitType.sample, epoch_size)
    mm_time_constant = [0]
    mm_schedule      = cntk.learner.momentum_as_time_constant_schedule(mm_time_constant, epoch_size)

    learner = cntk.learner.adam_sgd(out.parameters, lr = lr_schedule, momentum = mm_schedule)
    trainer = cntk.Trainer(out, ce, pe, learner)

    cntk.utils.log_number_of_parameters(out) ; print()
    progress_printer = cntk.utils.ProgressPrinter(tag = 'Training')

    rd.read_data_()
    val_data, val_label = rd.get_val()

    step = 1
    while step * minibatch_size < epoch_size:
        features, labels = rd.next_train_batch(minibatch_size, step - 1)
        trainer.train_minibatch({input_var: features, label_var: labels})
        step += 1
        progress_printer.update_with_trainer(trainer, with_metric = True)
        if step % 30 == 0:
            sys.stderr.write("feature shape" + str(features.shape) + '\n')
            print(step, "Training Accuracy:", 1 - trainer.test_minibatch({input_var: features, label_var: labels}))
            print(step, "Testing Accuracy:", 1 - trainer.test_minibatch({input_var: val_data, label_var: val_label}))

    progress_printer.epoch_summary(with_metric = True)
    print("Final Testing Accuracy:", 1 - trainer.test_minibatch({input_var: val_data, label_var: val_label}))

if __name__ == '__main__':
    net()
