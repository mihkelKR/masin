import numpy as np
import matplotlib.pyplot as plt
import bayesianfunctions2 as bf
import testfunctions as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time
from numpy import random


def runfunction(exploration,no_startingPoints,tavalist,suurem):
    """
    -------------
    Hyperhyperparameetrid
    ------------
    """
    a=np.random.randint(1,9)
    b=np.random.randint(1,1000)
    #bounds within which the optimiser is searching for the answer
    bounds = np.array([[-100.0, 100.0],[-100.0,100.0]])
    #number of measurements on the function
    n_iterations=200
    #exploration coefficient
 
    #how many random points the optimiser will try before deciding on the point with best expected improvement
    samplePoints=10000
    #how many random measurements of the function will be done before calculating expected improvement

    #algpunktide genereerimine

    X=np.random.uniform(bounds[0,0], bounds[0,1], [no_startingPoints,1])
    Y=np.random.uniform(bounds[1,0], bounds[1,1], [no_startingPoints,1])
    XY=np.column_stack((X,Y))
    Z=tf.rosenbrock(XY,a,b)


    #Mis mudel ja Kernel
    customKernel=C(1.0) * Matern()
    model = GaussianProcessRegressor(kernel=customKernel)


    for i in range(n_iterations):
        #print iteration
        print("Iteration " + str(i) + "/"+str(n_iterations))

        # Update Gaussian process with existing samples
        model.fit(XY, Z)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        XY_next = bf.propose_location(bf.expected_improvement, XY, Z, model, bounds, samplePoints, tavalist,suurem, no_startingPoints,exploration)
        
        # Obtain next sample from the objective function
        Z_next = tf.rosenbrock(XY_next,a,b)
        

        if Z_next>10:
            model = GaussianProcessRegressor(kernel=customKernel)
        else:
            model= GaussianProcessRegressor()
        
        # Add sample to previous samples
        XY = np.vstack((XY, XY_next))
        Z = np.vstack((Z, Z_next))

    minindex=np.argmin(Z)

    return Z[minindex]

