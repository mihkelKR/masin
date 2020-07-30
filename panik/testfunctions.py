import numpy as np
from numpy import random


def rosenbrock(XY, functionConstants):

    '''
    Evaluates the Rosenbrock function with given constants.

    Parameters
    ----------
    XY : numpy array
        (starting_points x 2)numpy array .
    functionConstants: list
        (list with rosenbrock constants [a,b]

    Returns
    -------
    Z: numpy array (1 x 1)
       Returns array with the function value evaluated at each XY point 
    '''

    x=XY[:,0]
    y=XY[:,1]
    Z=(functionConstants[0]-x)**2+functionConstants[1]*(y-x**2)**2
    return Z.reshape(-1,1)


def custom_function(XY, functionConstants):

    '''
    Evaluates the any given function with constants.

    Parameters
    ----------
    XY : numpy array
        (starting_points x 2)numpy array.
    functionConstants: list
        list with appropriate constants for the function

    Returns
    -------
    Z: numpy array
       Returns array with the function value evaluated at each XY point 
    '''

    x=XY[:,0]
    y=XY[:,1]

    """
    Z= INSERT 2D FUNCTION HERE and delete placeholder.
    """
    Z=x**2+y**2+functionConstants[0]

    return Z.reshape(-1,1)


def XDfunction(XD, functionConstants):

    '''
    Evaluates a function in X dimensions.

    Parameters
    ----------
    XD : numpy array
        (starting_points x X)numpy array.
    functionConstants: list
        list with appropriate constants for the function

    Returns
    -------
    Z: numpy array (starting_points x 1)
       Returns array with the function value evaluated at each XY point 

    '''
    D=XD.shape[1]
    Z=constants[0]
    for i in range(D):
        xi=XD[:,i]
        Z=Z+xi**2

    return Z

