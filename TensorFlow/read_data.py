import os
import numpy as np
import PIL.Image
from sklearn import preprocessing
import json
import math
global train_data, train_label, val_data, val_label, data_1, data_0
train_data, train_label, val_data, val_label, data_1, data_0 = [], [], [], [], [], []
ppp = -1
train_data_size = 0
val_data_size = 0
data_length = 36

def junk(frequency, energy, mean):
	wave = []
	time = []
	y0 = 0
	for i in xrange(len(frequency)):
		x0 = 0.02 * i
        #20ms per
        A = energy[i] / 100
        if(A == 0):
            for p in xrange(10):
                y0 = 0
                wave.append(y0)
                time.append(x0+p*0.001)
        else:
            phi = frequency[i] * 2.0 * math.pi
            b = math.asin(max(-1, min(y0/A, 1))) - phi * x0

            for p in xrange(10):
                y0 = A*math.sin(phi*(x0+p*0.001)+b)
                wave.append(y0)
                time.append(x0+p*0.001)
	fft_vals = np.fft.rfft(wave[1:-1])
	if np.mean(fft_vals) < mean * 4:
		return True
	else:
		return False
def readFile(filename, ind, type_):
    global ppp
    data_, num1, num0, num1_, num0_ = [], [], [], [], []
    print 'file start'
    fopen = open(filename, 'r')
    for eachLine in fopen:
        num0.append(float(eachLine))
    fopen.close()
    
    fopen = open(filename[0 : ind + 1] + 'f0', 'r')
    for eachLine in fopen:
        num1.append(float(eachLine))
    fopen.close()
    frequency = np.array(num1)
    energy = np.array(num0)
    
    
    wave = []
    time = []
    y0 = 0
    for i in xrange(len(frequency)):
        x0 = 0.02 * i
        #20ms per
        A = energy[i] / 100
        if(A == 0):
            for p in xrange(10):
                y0 = 0
                wave.append(y0)
                time.append(x0+p*0.001)
        else:
            phi = frequency[i] * 2.0 * math.pi
            b = math.asin(max(-1, min(y0/A, 1))) - phi * x0

            for p in xrange(10):
                y0 = A*math.sin(phi*(x0+p*0.001)+b)
                wave.append(y0)
                time.append(x0+p*0.001)
    
    fft_vals = np.fft.rfft(wave)
    glo_mean = np.mean(fft_vals)
    
    splice = 30
    useful_freq = np.array([])
    useful_ener = np.array([])
    for i in xrange(splice):
    	num = len(frequency) / splice
    	freq = []
    	ener = []
    	for j in xrange(num):
    		freq.append(frequency[i * num + j])
    		ener.append(energy[i * num + j])
    	if(junk(freq, ener, glo_mean) == False):
    		useful_freq = np.concatenate((useful_freq, freq))
    		useful_ener = np.concatenate((useful_ener, ener))
    p = PIL.Image.fromarray(np.array(useful_freq).reshape(1,len(useful_freq)).astype(np.float))
    p = p.resize((data_length,1),PIL.Image.BICUBIC)
    data_0.append(p.getdata())
    
def eachFile1(filepath, type_):
    global train_label, val_label
    global train_data_size, val_data_size
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        ind = allDir.index('.')
        
        if(allDir[ind + 1 : len(allDir)] == 'engy'):
            label_ = int(allDir[ind - 1]) - 1
            
            if(type_ == 'train'):
            	train_data_size += 1
                train_label.append(label_)
            if(type_ == 'test_new'):
            	val_data_size += 1
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
    eachFile0('train')
    eachFile0('test_new')
    
    
    data_0 = np.array(data_0)
    tmp_ = np.zeros([len(train_label), 4])
    tmp_[np.arange(len(train_label)), train_label] = 1
    train_label = tmp_
    tmp_ = np.zeros([len(val_label), 4])
    tmp_[np.arange(len(val_label)), val_label] = 1
    val_label = tmp_

def next_train_batch(_size, iter_):
    global train_data, train_label
    max_iter = train_data_size / _size
    iter = iter_ % max_iter
    return data_0[iter*_size : (iter + 1)*_size, :].reshape((_size, data_0.shape[1], 1)), train_label[iter*_size : (iter + 1)*_size]
    
    
def get_val():
    global val_data, val_label
    return data_0[train_data_size : train_data_size+val_data_size,:].reshape((val_data_size, data_0.shape[1], 1)), val_label

