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
def junk(frequency, energy, mean, glo_var):
    wave = []
    time = []
    y0 = 0
    for i in xrange(len(frequency)):
        x0 = 0.02 * i
        #20ms per
        A = np.sqrt(energy[i]) / 10
        if(A == 0):
            for p in xrange(20):
                y0 = 0
                wave.append(y0)
                time.append(x0+p*0.001)
        else:
            phi = frequency[i] * 2.0 * math.pi
            b = math.asin(max(-1, min(y0/A, 1))) - phi * x0

            for p in xrange(20):
                y0 = A*math.sin(phi*(x0+p*0.001)+b)
                wave.append(y0)
                time.append(x0+p*0.001)
    fft_vals = np.abs(np.fft.fft(wave[1:-1]))
    fft_vals = fft_vals / (len(fft_vals))
    if np.mean(fft_vals) < mean * 1.06:
    #if np.var(energy) > glo_var * 2:
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

def basic_noise_reduction(freq, ener):
    break_points = []
    avg = np.abs(np.mean(np.sqrt(np.abs(np.diff(freq, 1)))))
    prev = False
    for i in xrange(len(freq) - 1):
        curLen = np.sqrt((freq[i + 1] - freq[i]) ** 2)
        if(curLen > avg * 7):
            if prev:
                continue
            else:
                break_points.append(i + 1)
                prev = True
        else:
            prev = False
    '''
    diff = np.abs(np.diff(freq, 1))
    var = np.var(diff)
    dVar = []
    for i in xrange(len(diff)):
        new_var = np.var(np.concatenate((diff[0:i], diff[i+1:])))
        dVar.append(var - new_var)
    meanDvar = np.mean(dVar)
    for i in xrange(len(diff)):
        if(dVar[i] > 20 * meanDvar):
            break_points.append(i + 1)
    '''
    retA = []
    retB = []
    break_points = [0] + break_points
    break_points = break_points + [len(freq) - 1]
    print 'breaks ' + str(len(break_points))
    removed = [True for i in xrange(len(break_points) - 1)]
    retA = []
    retB = []
    '''if(len(break_points) <= 21):
        mxVal = 0
        mxrem = []
        for i in xrange(2**(len(break_points)-1)):
            removed_exp = []
            for j in xrange(len(break_points) - 1):
                if ((i >> j) & 1) == 1:
                    removed_exp.append(True)
                else:
                    removed_exp.append(False)
            prev = True
            flag = True
            for j in removed_exp:
                if(prev == False and j == False):
                    flag = False
                    break
                prev = j
            if(flag == False):
                continue
            else:
                tmpVal = 0
                for j in xrange(len(break_points) - 1):
                    if(removed_exp[j] == False):
                        tmpVal += break_points[j + 1] - break_points[j]
                if(tmpVal > mxVal):
                    mxVal = tmpVal
                    mxrem = removed_exp
        removed = mxrem
        print removed
    else:
        for i in xrange(len(break_points) - 2):
            ThisLen = break_points[i + 1] - break_points[i]
            AfterLen = break_points[i + 2] - break_points[i + 1]
            if(ThisLen > AfterLen):
                removed[i + 1] = True
            elif(i == 0):
                removed[i] = True
    '''
    
    f = [[[[2 ** 100, [], False] for k in xrange(len(freq) + 1)] for j in xrange(len(freq) + 1)] for i in xrange(len(freq))]

    for i in xrange(len(freq)):
        f[i][0][0][0] = 0
        f[i][0][0][1] = []
        f[i][0][0][2] = True
    f[0][1][1][0] = 0
    f[0][1][1][1] = [0]
    f[0][1][1][2] = True


    for i in xrange(0, len(freq) - 1):
        for j in xrange(0, i + 1):
            for k in xrange(0, i + 2):
                #f[i][j][k] -> f[i+1][j][k] or f[i][j][k] -> f[i + 1][j + 1][i + 1]
                if f[i][j][k][2] == False:
                    continue
                if f[i + 1][j][k][2]:
                    if(f[i][j][k][0] < f[i + 1][j][k][0]):
                        f[i + 1][j][k][0] = f[i][j][k][0]
                        f[i + 1][j][k][1] = f[i][j][k][1]
                else:
                    f[i + 1][j][k][0] = f[i][j][k][0]
                    f[i + 1][j][k][1] = f[i][j][k][1]
                    f[i + 1][j][k][2] = True
                if f[i + 1][j + 1][i + 1][2]:
                    b = freq[i]
                    if k != 0:
                        b = freq[k - 1]
                    if(np.abs(freq[i] - b) > 70):
                        b = -10000000
                    if (f[i + 1][j + 1][i + 1][0] > f[i][j][k][0] + (freq[i] - b) ** 2 + (i - (k - 1)) ** 1.5):
                        f[i + 1][j + 1][i + 1][0] = f[i][j][k][0] +(freq[i] - b) ** 2 + (i - (k - 1)) ** 1.5
                        f[i + 1][j + 1][i + 1][1] = f[i][j][k][1] + [i]
                else:
                    b = freq[i]
                    if k != 0:
                        b = freq[k - 1]
                    if(np.abs(freq[i] - b) > 70):
                        b = -10000000
                    f[i + 1][j + 1][i + 1][0] = f[i][j][k][0] + (freq[i] - b) ** 2 + (i - (k - 1)) ** 1.5
                    f[i + 1][j + 1][i + 1][1] = f[i][j][k][1] + [i]
                    f[i + 1][j + 1][i + 1][2] = True
    needed = len(freq) / 2 + 1
    minVal = 2 ** 100
    not_removed = []
    lb = 0
    for needed in xrange(len(freq) / 3, len(freq) + 1):
        for k in xrange(1, len(freq) + 1):
            #print 'True value for ', len(freq) - 1, needed, k, f[len(freq) - 1][needed][k][2]
            #print f[len(freq) - 1][needed][k][0]
            if(f[len(freq) - 1][needed][k][2] and minVal > float(f[len(freq) - 1][needed][k][0]) / (needed ** 6)):
                minVal = float(f[len(freq) - 1][needed][k][0]) / (needed ** 6)
                not_removed = f[len(freq) - 1][needed][k][1]

    retA = []
    retB = []
    for x in not_removed:
        retA.append(freq[x])
        retB.append(ener[x])
    last = not_removed[len(not_removed) - 1]
    flag = True
    for t in xrange(last, len(freq) - 1):
        if(freq[t] > freq[t + 1] and freq[t + 1] - freq[t] < 15):
            flag = False
            break
    if flag:
        for t in xrange(last + 1, len(freq) - 1):
            retA.append(freq[t])
            retB.append(ener[t])
    '''
    for i in xrange(len(break_points) - 1):
        l = break_points[i]
        r = break_points[i + 1] - 1
        if(removed[i] == False):
            new_X = new_X + x_axis[l: r + 1]
            retA = retA + freq[l:r+1]
            retB = retB + ener[l:r+1]
    '''
    filter_size = 1
    retA = filters.gaussian_filter(retA, filter_size)
    return retA, retB

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
    
    new_frequency = []
    new_energy = []
    flag = True
    for x in xrange(len(frequency)):
        if (frequency[x] == 0 or energy[x] < 300) and flag:
            continue
        flag = False
        new_frequency.append(frequency[x])
        new_energy.append(energy[x])
    frequency = (new_frequency)
    energy = (new_energy)
    new_frequency = []
    new_energy = []
    flag = True
    for x in reversed(range(len(frequency))):
        if (frequency[x] == 0 or energy[x] < 300) and flag:
            continue 
        flag = False
        new_frequency = ([frequency[x]]) + new_frequency
        new_energy = ([energy[x]]) + new_energy
    frequency = new_frequency
    energy = new_energy
    frequency, energy = basic_noise_reduction(frequency, energy)
    print len(frequency), filename
    wave = []
    time = []
    y0 = 0
    for i in xrange(len(frequency)):
        x0 = 0.02 * i
        #20ms per
        A = np.sqrt(energy[i]) / 10
        if(A == 0):
            for p in xrange(20):
                y0 = 0
                wave.append(y0)
                time.append(x0+p*0.001)
        else:
            phi = frequency[i] * 2.0 * math.pi
            b = math.asin(max(-1, min(y0/A, 1))) - phi * x0

            for p in xrange(20):
                y0 = A*math.sin(phi*(x0+p*0.001)+b)
                wave.append(y0)
                time.append(x0+p*0.001)
    
    fft_vals = np.abs(np.fft.fft(wave))
    fft_vals = fft_vals / (len(fft_vals))
    splice = 5

    useful_ener = np.array([])
    useful_data = [[] for t in xrange(n_channels)]
    useful_data[0] = frequency

    for i in xrange(splice):
        num = len(frequency) / splice
        freq = []
        ener = []
        for j in xrange(num):
            freq.append(frequency[i * num + j])
            ener.append(energy[i * num + j])
        fft_vals, state = junk(freq, ener, 0, 0)
        # Features
        Fs = 2000
        #useful_data[1].append(energyentropy(ener)) not useful
        C, S = stSpectralCentroidAndSpread(fft_vals, Fs)
        #useful_data[1].append(C)
        #useful_data[2].append(S)
        #useful_data[3].append(stSpectralEntropy(fft_vals))
        #useful_data[0].append(stSpectralEntropy(fft_vals))


    res_data = np.zeros((data_length, n_channels))
    for i in xrange(n_channels):
        p = PIL.Image.fromarray(np.array(useful_data[i]).reshape(1,len(useful_data[i])).astype(np.float))
        p = p.resize((data_length,1),PIL.Image.BICUBIC)
        res_data[:,i] = p.getdata()
    data_0.append(res_data)
    '''
    write = open('mydata/' + filename[2 : ind + 1] + 'f1', 'w')
    for x in res_data:
        write.write(str(x[0]) + '\n')
    write.close()
    '''
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

