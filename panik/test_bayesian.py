import pytest
import bayesianfunctions as bf
import testfunctions as tf
import numpy as np 

def test_testfunctionvalue():
    
    #do test function output correct values
	answer1=tf.rosenbrock(np.array([[1.0,1.0]]),1,100)
	assert answer1[0,0]==0.0

	answer2=tf.rosenbrock(np.array([[5.0,25.0]]),5,100)
	assert answer2[0,0]==0.0

	answer3=tf.sphere(np.array([[1.0,1.0]]),1,1)
	assert answer3[0,0]==2.0


def test_shape():

	answer4=tf.rosenbrock(np.array([[1.0,1.0]]),1,100)
	assert np.shape(answer4)==(1,1)

	
