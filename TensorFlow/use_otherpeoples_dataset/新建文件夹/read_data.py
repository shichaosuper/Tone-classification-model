import os
import numpy as np
import PIL.Image
from sklearn import preprocessing
import json
from scipy.ndimage import filters
global train_data, train_label, val_data, val_label, data_1, data_0
train_data, train_label, val_data, val_label, data_1, data_0 = [], [], [], [], [], []
ppp = -1
train_data_size = 400
val_data_size = 228
    

def read_data_():
    global train_data, train_label, val_data, val_label, data_1, data_0
    data_length = 36
    f = open("train.json")
    data = json.load(f)
    train_label = data['label']
    tmp = np.array(data['data'])[:,0,:]
    len_ = tmp.shape[1]
    data_0 = tmp
    tmp = np.array(data['data'])[:,1,:]
    data_1 = tmp
    f.close()
    f = open("test.json")
    data = json.load(f)
    val_label = data['label']
    tmp = np.array(data['data'])[:,0,:]
    data_0 = np.append(data_0, (tmp))
    data_0.resize(train_data_size + val_data_size, len_)
    tmp = np.array(data['data'])[:,1,:]
    data_1 = np.append(data_1, (tmp))
    data_1.resize(train_data_size + val_data_size, len_)
    f.close()
    print data_0.shape
    data_0 = np.array(data_1)
    data_0 = preprocessing.scale(data_0)
    data_1 = np.array(data_1)
    data_1 = preprocessing.scale(data_1)
    filter_size = 1
    for i in range(train_data_size):
        data = []
        g = filters.gaussian_filter(data_0[i], filter_size)
        p = PIL.Image.fromarray(g.reshape(1,len(data_0[i])).astype(np.float))
        p = p.resize((data_length,1),PIL.Image.BICUBIC)
        pp =p.getdata()
        data.append(np.array(pp,dtype='float'))
        
        g = filters.gaussian_filter(data_1[i], filter_size)
        p = PIL.Image.fromarray(g.reshape(1,len(data_1[i])).astype(np.float))
        p = p.resize((data_length,1),PIL.Image.BICUBIC)
        pp =p.getdata()
        data.append(np.array(pp,dtype='float'))
        train_data.append(np.transpose(data))
    for i in range(train_data_size, train_data_size + val_data_size):
        data = []
        g = filters.gaussian_filter(data_0[i], filter_size)
        p = PIL.Image.fromarray(g.reshape(1,len(data_0[i])).astype(np.float))
        p = p.resize((data_length,1),PIL.Image.BICUBIC)
        pp =p.getdata()
        data.append(np.array(pp,dtype='float'))
        g = filters.gaussian_filter(data_1[i], filter_size)
        p = PIL.Image.fromarray(g.reshape(1,len(data_1[i])).astype(np.float))
        p = p.resize((data_length,1),PIL.Image.BICUBIC)
        pp =p.getdata()
        data.append(np.array(pp,dtype='float'))
        val_data.append(np.transpose(data))
    train_data = np.array(train_data)
    
    tmp_ = np.zeros([len(train_label), 4])
    tmp_[np.arange(len(train_label)), train_label] = 1
    train_label = tmp_
    
    r = np.random.permutation(train_label.shape[0])
    train_data = train_data[r, :, :]
    train_label = train_label[r, :]
    print train_data.shape
    print train_label.shape

    

    val_data = np.array(val_data)
    
    tmp_ = np.zeros([len(val_label), 4])
    tmp_[np.arange(len(val_label)), val_label] = 1
    val_label = tmp_
    

    print val_data.shape
    print val_label.shape

def next_train_batch(_size, iter_):
    global train_data, train_label
    max_iter = train_label.shape[0] / _size
    iter = iter_ % max_iter
    if iter == 0:
        r = np.random.permutation(train_label.shape[0])
        train_data = train_data[r, :, :]
        train_label = train_label[r, :]
    return train_data[iter*_size : (iter + 1)*_size], train_label[iter*_size : (iter + 1)*_size]
    
    
def get_val():
    global val_data, val_label
    return val_data, val_label

