#! /usr/bin/env python3

import matplotlib.pyplot as plt 
import numpy as np 
import scipy

#lisab datale suvalisust
def addNoise(noise):
	return np.random.normal(loc=0, scale=noise)

#funktsiooni vaartuste arvutamine
def f(x, noise=0.1):
	y=np.zeros(len(x))
	for i in range(len(x)):
		y[i]=x[i]**2*np.sin(5*np.pi*x[i])**6 + addNoise(noise)
	return y 

"""
kohad=np.linspace(0,1,100)
vaartused=f(kohad)
vaartusedNoisy=f(kohad,0.1)

plt.scatter(kohad, vaartused, s=4, c='r')
plt.scatter(kohad, vaartusedNoisy, s=4, c='b')
plt.show()
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF 


mudel = GaussianProcessRegressor()

#olemasolevate andmete p6hjal arvutab mudeli
def fitModel(model, x):
	y=f(x,0.05)
	x=x.reshape(len(x),1)
	y=y.reshape(len(y),1)
	return model.fit(x,y)

def predictValue(model, x):
	x=x.reshape(len(x),1)
	return model.predict(x)

def predictStdAndValue(model, x):
	x=x.reshape(len(x),1)
	return model.predict(x, return_std=True)

#plots the real data vs surrogate function
def RealVSsurrogate(model,x):
	#reaalne graafik
	y=f(x,0.05)
	plt.scatter(x,y, s=4, c='r')
	#surrogate graafik
	fitModel(model,x)
	Xsample=np.asarray(np.arange(0,1,0.001))
	y_surrogate= predictValue(model, Xsample)
	plt.plot(Xsample, y_surrogate)
	# show the plot
	plt.show()


def expected_improvement(model, sampleArray,  points, exploration = 0.01):

	#arvutab puntki v22rtuse
	mean, std = predictStdAndValue(model,points)
	#arvutab praegu olemasolevate punktide vaartused ning leiab neist parima
	functionValues=predictValue(model,sampleArray)
	bestThusFar = np.max(functionValues)

	std = std.reshape(len(std),1)
	EI=np.zeros(len(mean))
	print(points)
	print(mean)
	print(std)

	#matemaatika, mis arvutab Expected improvementi valmis
	with np.errstate(divide = 'warn'):
		for i in range(len(mean)):
			print(mean[i])
			preint(std[i])
			if std[i] == 0.0:
				EI[i] = 0.0
			else:
				improvement = (mean[i] - bestThusFar - exploration)
				Z = improvement / std[i]
				EI = improvement * scipy.stats.norm.cdf(Z)+ std*scipy.stats.norm.pdf(Z)
		
	return EI 

suvakad=np.random.rand(10)
RealVSsurrogate(mudel,suvakad)
fitModel(mudel,suvakad)
print(expected_improvement(mudel, suvakad, np.linspace(0,1,10)))





