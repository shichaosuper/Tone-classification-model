import matplotlib
import matplotlib.pyplot as plt

fx = open('080801_huweif_xiang1.engy')
fy = open('080801_huweif_xiang1.f0')

x = []
y1 = []
y2 = []
a = 0
while True:
	try:
		xx = float(fx.next())
		yy = float(fy.next())
		x.append(a)
		y1.append(xx)
		y2.append(yy)
		a = a + 1
	except:
		fx.close()
		fy.close()
		break

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()