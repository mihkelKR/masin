import numpy as np
import matplotlib.pyplot as plt
import bayesianfunctions2 as bf
import testfunctions as fu
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time


def run(evaluateFunction, functionConstants, n_iterations,bounds=np.array([[-100.0, 100.0],[-100.0,100.0]])):

    '''
    Runs the Bayesian optimiser on any given function in mihkelBayesian.functions

    Parameters
    ----------
    evaluateFunction : string
        String corresponding to the function name that is being evaluated. 
    functionConstants: list
        List with any relevant constants corresponding to evaluateFunction
    n_iternations : integer
        How many measurements of the function to reach the minimum.
    boundaries: numpy array (2 x 2)
        Boundaries of the searchspace in the form of [[xmin,xmax],[ymin,ymax]]

    Returns
    -------
    min_val: float
        Smallest evaluation.
    min_point: numpy array (1 x 2)
        Point correspoinding to the smallest evaluation
    '''


    #Exploitation-exploration trade-off parameter
    exploration=100
    #Number of random points considered for expected improvement
    samplepoints=10000
    #Number of random measurements done before applying Bayesian optimisation
    no_startingpoints=int(n_iterations/5)
    #Number of evaluation before applying restricted boundries
    iterations_before_constraints=int(n_iterations/5*2)
    #How much larger can boundaries be from minimum point
    larger=5000
    #Generating startingpoints

    XY,Z=bf.generate_startingArrays(bounds,no_startingpoints, evaluateFunction, functionConstants)

    #Which Gaussian model
    customKernel=C(1.0) * Matern()
    model = GaussianProcessRegressor(kernel=customKernel)


    for i in range(n_iterations-no_startingpoints):
        #print iteration
        print("Iteration " + str(i + no_startingpoints) + "/"+str(n_iterations))

        # Update Gaussian process with existing samples
        model.fit(XY, Z)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        XY_next = bf.propose_location(bf.expected_improvement, XY, Z, model, bounds, samplepoints, iterations_before_constraints,larger, no_startingpoints,exploration)
        
        
        # Obtain next sample from the objective function
        Z_next = eval("fu."+evaluateFunction)(XY_next,functionConstants)
        
        #Changes kernel for very small values

        if Z_next>10:
            model = GaussianProcessRegressor(kernel=customKernel)
        else:
            model= GaussianProcessRegressor()
        
        # Add sample to previous samples
        XY = np.vstack((XY, XY_next))
        Z = np.vstack((Z, Z_next))


    minindex=np.argmin(Z)  
    min_val=Z[minindex]
    min_point=XY[minindex]  
    print(min_val,min_point)

    return min_val, min_point


