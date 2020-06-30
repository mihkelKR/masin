#! /usr/bin/env python3

import tensorflow as tf 
from tensorflow import keras
import numpy as np  
import matplotlib.pyplot as plt 

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images=tf.keras.utils.normalize(train_images, axis=1)
test_images=tf.keras.utils.normalize(test_images, axis=1)

#print(train_images[0])
#plt.imshow(train_images[0])
#plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images,train_labels, epochs=3)

loss, acc=model.evaluate(test_images,test_labels)


print ("loss is",loss," and acc is",acc)

model.save('mihkliesimenemudel.model')


