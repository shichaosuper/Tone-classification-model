from layer_utils import *
import data_loader
import theano.tensor as T
import numpy
import theano
import collections
class model(object):

    def run_net(self, learning_rate, training_iters, batch_size, display_step, n_input, n_classes, dropout, n_channel):
        """ Demonstrates lenet on MNIST dataset
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """

        datasets = data_loader.load_data()

        train_set_x, train_set_y, train_data_size, train_set_y_extend = theano.shared(numpy.array(datasets[0][0])), theano.shared(numpy.array(datasets[0][1])), datasets[0][2], theano.shared(datasets[0][3])
        test_set_x, test_set_y = theano.shared(numpy.array(datasets[1][0])), theano.shared(numpy.array(datasets[1][1]))

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = theano.tensor.tensor4('x')   # the data is presented as rasterized images
        y = theano.tensor.vector('y', dtype='int64')  # the labels are presented as 1D vector of
                            # [int] labels
        y_extend = theano.tensor.dmatrix('y_extend')

        ishape = (n_input)  # this is the size of MNIST images

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        cv_layer0_input = x

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        cv_layer0 = Conv_Relu_Pool_Layer(input=cv_layer0_input,
                filter_shape=(64, 1, 3, 1), pool_size=(2, 1), border_mode=(2,0))


        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        cv_layer1 = Conv_Relu_Pool_Layer(input=cv_layer0.output,
                filter_shape=(72, 64, 3, 1), pool_size=(2, 1),border_mode=(2,0))

        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)

        cv_layer2 = Conv_Relu_Pool_Layer(input = cv_layer1.output, filter_shape=(96, 72, 3, 1), pool_size=(2, 1),border_mode=(2,0))


        fc_layer0_input = cv_layer2.output.flatten(2)

        # construct a fully-connected layer
        fc_layer0 = fc_layer(input=fc_layer0_input, n_in=96,n_out=96, dropout_prob=dropout)
        fc_layer1 = fc_layer(input=fc_layer0.output, n_in=96, n_out=72, dropout_prob=dropout)
        fc_layer2 = fc_layer(input=fc_layer1.output, n_in=72, n_out=64, dropout_prob=dropout)

        #to output
        out = fc_layer(input=fc_layer2.output, n_in=64, n_out=n_classes, dropout_prob=1)
        pyx = theano.tensor.nnet.softmax(out.output)
        cost = T.mean(theano.tensor.nnet.nnet.categorical_crossentropy(pyx, y_extend))
        pred = T.argmax(pyx, axis = 1)
        # the cost we minimize during training is the NLL of the model
        # create a function to compute the mistakes that are made by the model
        test_model = theano.function([], T.mean(T.eq(pred, y)),
                 givens={
                    x: test_set_x,
                    y: test_set_y})

        # create a list of all model parameters to be fit by gradient descent
        params = fc_layer2.params + fc_layer1.params + fc_layer0.params + cv_layer2.params + cv_layer1.params + cv_layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        '''
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates dictionary by automatically looping over all
        # (params[i],grads[i]) pairs.
        
        '''
        updates = collections.OrderedDict()
        for param_i, grad_i in zip(params, grads):
            updates[param_i] = param_i - learning_rate * grad_i
        
        #t = theano.shared(np.float32(1))
        #updates = adam(cost, params, learning_rate, t)
        train_model = theano.function([index], [cost, T.mean(T.eq(pred, y))], updates=updates,
              givens={
                x: train_set_x[(index % (train_data_size / batch_size)) * batch_size: ((index) % (train_data_size / batch_size) + 1) * batch_size],
                y: train_set_y[(index % (train_data_size / batch_size)) * batch_size: ((index) % (train_data_size / batch_size) + 1) * batch_size],
                y_extend: train_set_y_extend[(index % (train_data_size / batch_size)) * batch_size: ((index) % (train_data_size / batch_size) + 1) * batch_size]})

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        '''
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
        
        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False
        '''
        step = 1
        max_ = 0.1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            # Run optimization op (backprop)
            #loss, acc_ = sess.run([merged,optimizer], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            res = train_model(step)
            step += 1
            theano.printing.Print(pred)
            if step % display_step == 0:
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6}".format(res[0]) + ", Training Accuracy= " + \
                      "{:.5}".format(res[1]))
            if(step % 6 == 0):
                num_ = test_model()
                if (num_ > max_):
                    max_ = num_
                print("Testing Accuracy:", num_)
        print("Optimization Finished!")
        print("max acc:", max_)
        
        
        # Calculate accuracy for 300 mnist test images
        testAcc = test_model()
        print("Testing Accuracy:", testAcc)
myModel = model()
myModel.run_net(learning_rate = 0.001, training_iters = 60000, batch_size = 21, display_step = 100, n_input = 40, n_classes = 4, dropout = 0.9, n_channel = 1)
