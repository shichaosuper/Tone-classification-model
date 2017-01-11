import os
import numpy as np
import numpy
import PIL.Image
from sklearn import preprocessing
import json
import math
global train_data, train_label, val_data, val_label, data_1, data_0
train_data, train_label, val_data, val_label, data_1, data_0 = [], [], [], [], [], []
ppp = -1
n_channels = 6
train_data_size = 0
val_data_size = 0
data_length = 36
eps = 1e-8
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
	fft_vals = np.abs(np.fft.fft(wave[1:-1]))
        fft_vals = fft_vals / (len(fft_vals))
	if np.mean(fft_vals) < mean * 1.5:
		return fft_vals, True
	else:
		return fft_vals, False
def energyentropy(ener):
    Eol = np.sum(ener)
    ener = np.array(ener) / Eol
    return -np.sum(ener * np.log2(ener+1e-8))
def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))
    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)
def stSpectralEntropy(X, numOfShortBlocks=3):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En
def readFile(filename, ind, type_):
    global ppp, n_channels
    data_, num1, num0, num1_, num0_ = [], [], [], [], []
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
    
    fft_vals = np.abs(np.fft.fft(wave))
    fft_vals = fft_vals / (len(fft_vals))
    glo_mean = np.mean(fft_vals)
    splice = 30
    useful_ener = np.array([])
    useful_data = [[] for t in xrange(n_channels)]
    for i in xrange(splice):
    	num = len(frequency) / splice
    	freq = []
    	ener = []
    	for j in xrange(num):
    	    freq.append(frequency[i * num + j])
    	    ener.append(energy[i * num + j])
        fft_vals, state = junk(freq, ener, glo_mean)
	if(state == False):
        # Features
	    Fs = 1000
            useful_data[0].append(np.mean(freq))
            useful_data[1].append(np.max(fft_vals))
            useful_data[2].append(energyentropy(ener))
            C, S = stSpectralCentroidAndSpread(fft_vals, Fs)
            useful_data[3].append(C)
            useful_data[4].append(S)
            useful_data[5].append(stSpectralEntropy(fft_vals))
            
    res_data = np.zeros((data_length, n_channels))
    for i in xrange(n_channels):
        p = PIL.Image.fromarray(np.array(useful_data[i]).reshape(1,len(useful_data[i])).astype(np.float))
        p = p.resize((data_length,1),PIL.Image.BICUBIC)
        res_data[:,i] = p.getdata()
    data_0.append(res_data)
    
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
    global train_data, train_label, n_channels
    max_iter = train_data_size / _size
    iter = iter_ % max_iter
    return data_0[iter*_size : (iter + 1)*_size, :].reshape((_size, data_0.shape[1], n_channels)), train_label[iter*_size : (iter + 1)*_size]
    
    
def get_val():
    global val_data, val_label, n_channels
    return data_0[train_data_size : train_data_size+val_data_size,:].reshape((val_data_size, data_0.shape[1], n_channels)), val_label

