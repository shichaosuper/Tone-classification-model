import matplotlib.pyplot as plt
import numpy as np
import os
import wave
import math

def readfile(path):
	f = open(path, 'r')
	ret = []
	while True:
		try:
			x = float(f.next())
			ret.append(x)
		except:
			break
	
	f.close()
	return ret

files = os.listdir('.')
frequency = dict()
energy = dict()
fft = []

wav = []

count = 0
for FILE in files:
	if(FILE[-3:] == '.f0'):
		t = readfile(FILE)
		if(frequency.has_key(FILE[:-3]) == False):
			frequency[FILE[:-3]] = []
		frequency[FILE[:-3]] = (t)
	else:
		t = readfile(FILE)
		if(energy.has_key(FILE[:-5]) == False):
			energy[FILE[:-5]] = []
		energy[FILE[:-5]] = (t)

time = []

Target = np.random.randint(0, len(frequency) - 1)
count = 0
for k in frequency:
	y0 = 0.0
	if(count < Target):
		count = count + 1
		continue
	for i in xrange(len(frequency[k])):
		x0 = 0.02 * i
		#20ms per
		A = energy[k][i] / 100
		if(A == 0):
			for p in xrange(20):
				y0 = 0
				wav.append(y0)
				time.append(x0+p*0.001)
		else:
			phi = frequency[k][i] * 2.0 * math.pi
			b = math.asin(max(-1, min(y0/A, 1))) - phi * x0

			for p in xrange(20):
				y0 = A*math.sin(phi*(x0+p*0.001)+b)
				wav.append(y0)
				time.append(x0+p*0.001)
	break

f, arr = plt.subplots(1, 2)
arr[0].plot(time, wav)
arr[1].plot(np.array(time) / len(time), np.abs(np.fft.fft(wav)), '*')
plt.show()