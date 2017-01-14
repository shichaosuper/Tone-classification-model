import numpy as np
import matplotlib.pyplot as plt





import os
import numpy as np
import PIL.Image
from sklearn import preprocessing
import json
from scipy.ndimage import filters
global train_data, train_label, val_data, val_label
global data_1_train, data_0_train
global data_1_val, data_0_val
train_data, train_label, val_data, val_label, data_1_train, data_0_train, data_1_val, data_0_val = [], [], [], [], [], [], [], []
ppp = -1
train_data_size = 400
val_data_size = 228
    
def readFile(filename, ind, type_):
    global ppp
    num1_, num0_ = [], []
    fopen = open(filename, 'r')
    num0 = fopen.readlines()
    fopen.close()
    
    fopen = open(filename[0 : ind + 1] + 'f0', 'r')
    num1 = fopen.readlines()
    fopen.close()
    
    f_list0 = map(np.float,num0)
    f_list1 = map(np.float,num1)
    start_ = 0
    end_ = len(f_list0) - 1
    while(f_list0[start_] < min(np.max(f_list0), 700)):
      start_+=1
    while(f_list0[end_] < 300):
      end_ -= 1
    f_list0 = f_list0[start_ : end_]
    f_list1 = f_list1[start_ : end_]
    x = np.linspace(0,1,len(f_list0))
    y0 = np.array(f_list0)
    y1 = np.array(f_list1)
    cof0 = np.polyfit(x,y0,9)
    cof1 = np.polyfit(x,y1,9)
    p0=np.poly1d(cof0)
    p1=np.poly1d(cof1)
    num0_ = p0(x)
    num1_ = y1
    p = PIL.Image.fromarray(np.array(num1_).reshape(1,len(num1_)).astype(np.float))
    p = p.resize((128,1),PIL.Image.BICUBIC)
    pp =p.getdata()
    num1_ = np.array(pp,dtype='float')
    p = PIL.Image.fromarray(np.array(num0_).reshape(1,len(num0_)).astype(np.float))
    p = p.resize((128,1),PIL.Image.BICUBIC)
    pp =p.getdata()
    num0_ = np.array(pp,dtype='float')
    if(type_ == 'train'):
        data_0_train.append(map(float, num0_))
        data_1_train.append(map(float, num1_))
    else:
        data_0_val.append(map(float, num0_))
        data_1_val.append(map(float, num1_))


        
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
        
def store(data, s):
    with open(s, 'w') as json_file:
        json_file.write(json.dumps(data, skipkeys = True))
        json_file.close()

def default(obj):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return obj.tolist()
        else:
            return [default(obj[i]) for i in range(obj.shape[0])]
        
def read_data_():
    global train_data, train_label, val_data, val_label, data_1, data_0
    global data_1_train, data_0_train
    global data_1_val, data_0_val
    eachFile0('train')
    eachFile0('test_new')
    data_0_train = np.array(data_0_train)
    data_0_val = np.array(data_0_val)
    data_1_train = np.array(data_1_train)
    data_1_val = np.array(data_1_val)
    data = np.zeros((data_0_train.shape[0], 2, data_0_train.shape[1]))
    data[:, 0, :] = data_0_train
    data[:, 1, :] = data_1_train
   
    data_p = {}
    data_p['data'] = default(data)
    data_p['label'] = train_label 
    store(data_p, 'train.json')
    
    data = np.zeros((data_0_val.shape[0], 2, data_0_val.shape[1]))
    data[:, 0, :] = data_0_val
    data[:, 1, :] = data_1_val
    data_p = {}
    data_p['data'] = default(data)
    data_p['label'] = val_label 
    store(data_p, 'test.json')
    
read_data_()