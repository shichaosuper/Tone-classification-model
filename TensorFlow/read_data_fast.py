import os
import numpy as np
import numpy
import PIL.Image
from sklearn import preprocessing
import json
import math
from scipy.ndimage import filters
global train_data, train_label, val_data, val_label, data_1, data_0
train_data, train_label, val_data, val_label, data_1, data_0 = [], [], [], [], [], []
ppp = -1
n_channels = 1
train_data_size = 0
val_data_size = 0
data_length = 40
eps = 1e-8

def readFile_processed(filename, ind, type):
    global ppp, n_channels
    data_, num1, num0, num1_, num0_ = [], [], [], [], []
    fopen = open(filename, 'r')
    for eachLine in fopen:
        num0.append(float(eachLine))
    fopen.close()
    
    num0 = np.array(num0)
    res_data = np.zeros((data_length, n_channels))
    for i in xrange(n_channels):
        res_data[:, i] = num0
    data_0.append(res_data)
    
def eachFile1(filepath, type_):
    global train_label, val_label
    global train_data_size, val_data_size
    pathDir =  os.listdir(filepath)
    _ = ''
    for allDir in pathDir:
        #child = os.path.join('%s/%s' % (filepath, allDir))
        child = os.path.join('%s/%s' % (filepath, allDir))
        ind = allDir.index('.')
        
        #if(allDir[ind + 1 : len(allDir)] == 'engy'):
        if(allDir[ind + 1 : len(allDir)] == 'f1'):
            label_ = int(allDir[ind - 1]) - 1
            
            if(type_ == 'mydata/train'):
            	train_data_size += 1
                train_label.append(label_)
            if(type_ == 'mydata/test_new'):
            	val_data_size += 1
                val_label.append(label_)
            pron = allDir[0:ind - 1]
            #readFile(child, ind + len(filepath)+ 1, type_)
            print child
            readFile_processed(child, ind + len(filepath)+ 1, type_)
            
def eachFile0(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('./%s/%s' % (filepath, allDir))
        eachFile1(child, filepath)
        
def read_data_():
    global train_data, train_label, val_data, val_label, data_1, data_0    
    #eachFile0('train')
    #eachFile0('test_new')
    eachFile0('mydata/train')
    eachFile0('mydata/test_new')
    
    data_0 = np.array(data_0)
    tmp_ = np.zeros([len(train_label), 4])
    tmp_[np.arange(len(train_label)), train_label] = 1
    train_label = tmp_
    tmp_ = np.zeros([len(val_label), 4])
    tmp_[np.arange(len(val_label)), val_label] = 1
    val_label = tmp_

def next_train_batch(_size, iter_):
    global train_data, train_label, n_channels
    max_iter = train_data_size / _size
    iter = iter_ % max_iter
    return data_0[iter*_size : (iter + 1)*_size, :].reshape((_size, data_0.shape[1], n_channels)), train_label[iter*_size : (iter + 1)*_size]
    
    
def get_val():
    global val_data, val_label, n_channels
    return data_0[train_data_size : train_data_size+val_data_size,:].reshape((val_data_size, data_0.shape[1], n_channels)), val_label

