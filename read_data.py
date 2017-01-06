import os
import numpy as np
import PIL.Image

global train_data
train_data = []
global val_data
val_data = []
global train_label
train_label = []
global val_label
val_label = []
global t
t = -1

def readFile(filename, ind, type_):
    global train_data
    global val_data
    global train_label
    global val_label
    global t
    
    data_ = []
    fopen = open(filename, 'r')
    num1 = []
    num0 = []
    num1_ = []
    num0_ = []
    for eachLine in fopen:
        num1.append(eachLine)
    t = max(t, len(num1))
    fopen.close()
    
    fopen = open(filename[0 : ind + 1] + 'f0', 'r')
    for eachLine in fopen:
        num0.append(eachLine)
    fopen.close()
    
    len_ = len(num1)
    for i in range(len_):
        if(num1[i] != 0 and num0[i] != 0):
            num1_.append(num1[i])
            num0_.append(num0[i])
    
    p = PIL.Image.fromarray(np.array(num1_).reshape(1,len(num1_)).astype(np.float))
    p = p.resize((230,1),PIL.Image.BICUBIC)
    pp =p.getdata()
    num1_ = np.array(pp,dtype='float')
    p = PIL.Image.fromarray(np.array(num0_).reshape(1,len(num0_)).astype(np.float))
    p = p.resize((230,1),PIL.Image.BICUBIC)
    pp =p.getdata()
    num0_ = np.array(pp,dtype='float')
    data_.append(map(float, num1_))
    data_.append(map(float, num0_))
    
    
    data_ = np.transpose(data_)
    if(type_ == 'train'):
        train_data.append(data_)
    if(type_ == 'test_new'):
        val_data.append(data_)
    
    
def eachFile0(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('./%s/%s' % (filepath, allDir))
        eachFile1(child, filepath)
        
def eachFile1(filepath, type_):
    global train_data
    global val_data
    global train_label
    global val_label
    global t
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
        
def read_data_():
    global train_data
    global val_data
    global train_label
    global val_label
    global t
    eachFile0('test_new')
    eachFile0('train')
    len_x = len(train_data)
    
    for i in range(len_x):
        tmp = len(train_data[i])
        train_data[i] = list(train_data[i])
        train_data[i]+=[np.array([0,0]) for j in range(t - tmp)]
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    tmp_ = np.zeros([len(train_label), 4])
    tmp_[np.arange(len(train_label)), train_label] = 1
    train_label = tmp_
    r = np.random.permutation(len(train_label))
    train_data = train_data[r, :, :]
    train_label = train_label[r, :]
    print train_data.shape
    print train_label.shape

    
    len_x = len(val_data)
    for i in range(len_x):
        tmp = len(val_data[i])
        val_data[i] = list(val_data[i])
        val_data[i]+=[np.array([0,0]) for j in range(t - tmp)]
    val_data = np.array(val_data)
    val_label = np.array(val_label)
    tmp_ = np.zeros([len(val_label), 4])
    tmp_[np.arange(len(val_label)), val_label] = 1
    val_label = tmp_
    mean_image = np.mean(train_data, axis=0)
    mean_image_ = np.mean(val_data, axis=0)
    mean_image+=mean_image_
    mean_image/=2 
    train_data -= mean_image
    val_data -= mean_image
    print val_data.shape
    print val_label.shape
    print t

  
def next_train_batch(_size, iter_):
    global train_data
    global val_data
    global train_label
    global val_label
    global t
    max_iter = train_label.shape[0] / _size
    iter = iter_ % max_iter
    return train_data[iter*_size : (iter + 1)*_size], train_label[iter*_size : (iter + 1)*_size]
    
    
def get_val():
    global train_data
    global val_data
    global train_label
    global val_label
    global t
    return val_data, val_label

