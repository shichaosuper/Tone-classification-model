import os
import numpy as np
import PIL.Image
from sklearn import preprocessing
import json
global train_data, train_label, val_data, val_label, data_1, data_0
train_data, train_label, val_data, val_label, data_1, data_0 = [], [], [], [], [], []
ppp = -1
train_data_size = 400
val_data_size = 228
    
def readFile(filename, ind, type_):
    global ppp
    data_, num1, num0, num1_, num0_ = [], [], [], [], []
    fopen = open(filename, 'r')
    for eachLine in fopen:
        num0.append(eachLine)
    fopen.close()
    
    fopen = open(filename[0 : ind + 1] + 'f0', 'r')
    for eachLine in fopen:
        num1.append(eachLine)
    fopen.close()
    
    len_ = len(num1)
    for i in range(len_):
        if(num0[i] > 500.0 and num1[i] !=0):
            num0_.append(num0[i])
            num1_.append(num1[i])
    p = PIL.Image.fromarray(np.array(num1_).reshape(1,len(num1_)).astype(np.float))
    p = p.resize((128,1),PIL.Image.BICUBIC)
    pp =p.getdata()
    num1_ = np.array(pp,dtype='float')
    p = PIL.Image.fromarray(np.array(num0_).reshape(1,len(num0_)).astype(np.float))
    p = p.resize((128,1),PIL.Image.BICUBIC)
    pp =p.getdata()
    num0_ = np.array(pp,dtype='float')
    data_0.append(map(float, num0_))
    data_1.append(map(float, num1_))
    

        
def eachFile1(filepath, type_):
    global train_label, val_label
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        ind = allDir.index('.')
        
        if(allDir[ind + 1 : len(allDir)] == 'engy'):
            label_ = int(allDir[ind - 1]) -1
            if(type_ == 'train'):
                train_label.append(label_)
            if(type_ == 'test_new'):
                val_label.append(label_)
            pron = allDir[0:ind - 1]
            readFile(child, ind + len(filepath)+ 1, type_)
            
            
def eachFile0(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('./%s/%s' % (filepath, allDir))
        eachFile1(child, filepath)
        
def read_data_():
    global train_data, train_label, val_data, val_label, data_1, data_0
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
    #eachFile0('train')
    #eachFile0('test_new')
    
    
    data_0 = np.array(data_0)
    data_0 = preprocessing.scale(data_0)
    data_0 = preprocessing.scale(data_0)
    data_1 = np.array(data_1)
    #data_1 = preprocessing.scale(data_1)
    for i in range(train_data_size):
        data = []
        data.append(data_0[i])
        data.append(data_1[i])
        train_data.append(np.transpose(data))
    for i in range(train_data_size, train_data_size + val_data_size):
        data = []
        data.append(data_0[i])
        data.append(data_1[i])
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
    return train_data[iter*_size : (iter + 1)*_size], train_label[iter*_size : (iter + 1)*_size]
    
    
def get_val():
    global val_data, val_label
    return val_data, val_label

