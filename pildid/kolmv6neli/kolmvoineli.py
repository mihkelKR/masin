#! /usr/bin/env python3

import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow import keras 
import cv2
import os


DATADIR = "/home/mihkel/masin/pildid/kolmv6neli/"
CATEGORIES=["kolm", "neli"]

pildid=cv2.imread(os.path.join(DATADIR, CATEGORIES[0], "00000.png"), cv2.IMREAD_GRAYSCALE) #training array in correct format
labels=np.array([]) #label array

for category in CATEGORIES:

	PATH=os.path.join(DATADIR, category)
	for img in os.listdir(PATH): #creates a list of all images in the path

		img_array=cv2.imread(os.path.join(PATH,img), cv2.IMREAD_GRAYSCALE) #converts image to numy array
		pildid=np.dstack([pildid,img_array]) #adds array to training array
		if category=="kolm": #labels appropriately
			labels=np.append(labels,0)
		else:
			labels=np.append(labels,1)

print(type(pildid))
print(pildid.shape)
pildid.shape=(11974,28,28)
print(pildid.shape)
print(pildid[324])
