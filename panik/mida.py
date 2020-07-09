#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
import scipy 

#Funktsiooni graafik

x=np.linspace(0,1,200)

def addNoise(noise):
	return np.random.normal(loc=0, scale=noise)


def calcFunction(x, noise):
	y=np.zeros(len(x))
	for i in range(len(x)):
		y[i]=x[i]**2*np.sin(5*np.pi*x[i])**6 + addNoise(noise)
	return y 


y=calcFunction(x, 0)
ynoise=calcFunction(x,0.1)

"""

plt.scatter(x,y, s=4)
plt.show()
plt.scatter(x,ynoise,s=4)
plt.show()
print(np.argmax(y),np.argmax(ynoise))

"""

from sklearn.gaussian_process import GaussianProcessRegressor

#importime gaussian funktsiooni mis annab meile surrogaat funktsiooni
mudel = GaussianProcessRegressor()

#mudeli treenimine antud andmetega
def fitModel(a,b, model):
	a=a.reshape(len(a),1)
	b=b.reshape(len(b),1)
	return model.fit(a,b)


def predictValue(model, a):
	a=a.reshape(len(a),1)
	return model.predict(a)

def predictValueAndSTD(model, a):
	a=a.reshape(len(a),1)
	return model.predict(a, return_std=True)

def RealVSsurrogate(a,b,model):
	plt.scatter(a,b, s=4, c='r')
	Xsamples = np.asarray(np.arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples= predictValue(model, Xsamples)
	plt.plot(Xsamples, ysamples)
	# show the plot
	plt.show()


#ACQUISITION FUNCTION

def acquisition(x, Xsamples, model):

	yhat= predictValue(mudel,x)
	best=max(yhat)
	
	mean, std = predictValueAndSTD(mudel,x)
	#ei saa sellest osast h2sti aru aga okei
	mean=mean[:, 0]
	probs=scipy.stats.norm.cdf((mean-best)/(std+10**(-9)))
	return probs


def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = random(100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = argmax(scores)
	return Xsamples[ix, 0]