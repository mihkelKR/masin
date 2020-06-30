#! /usr/bin/env python3

import tensorflow as tf 
from tensorflow import keras
import numpy as np  
import matplotlib.pyplot as plt 

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = tf.keras.models.load_model('mihkliesimenemudel.model')

ennustus = model.predict(test_images)

print(ennustus[5])

plt.imshow(test_images[5])
plt.show()