import numpy as np
from scipy.stats import norm
import sklearn
import testfunctions as fu 



def generate_startingArrays(bounds,no_startingPoints, a, b):


    '''
    Generates random starting points within starting boundaries. 

    Parameters
    ----------
    bounds: numpy array (2 x 2)
        Bounds of applied on XY.
    no_startingPoints: integer
        Number of random evaluations before using the expected improvement.
    a : float
        constant a in the Rosenbrock funtion.
    b : float
        constant b in the Rosenbrock funtion.

    Returns
    -------
    XY: numpy array (no_startingpoints, 2)
        Numpy array of poistions that the function was evaluated at
    Z: numpy array (no_startingpoints, 1)
        Numpy array with values corresponding to XY

    '''

    X=np.random.uniform(bounds[0,0], bounds[0,1], [no_startingPoints,1])
    Y=np.random.uniform(bounds[1,0], bounds[1,1], [no_startingPoints,1])
    XY=np.column_stack((X,Y))
    Z=fu.rosenbrock(XY,a,b)

    return XY,Z


def expected_improvement(pointsConsidered, XY, model, exploration):
    
    '''
    Computes the expected improvement at considered points based on existing Gaussian process surrogate model.

    Parameters
    ----------
    pointsConsidered: numpy array (n x 2)
        Points at which EI shall be computed.
    XY: numpy array (m x 2)
        Evaluated point locations.
    model: sklearn.gaussian_process
        A GaussianProcessRegressor fitted to points in XY and their evaluations Z.
    exlporation: float
        Exploitation-exploration trade-off parameter.

    Returns
    -------
    ei: numpy array
       Expected improvements at considered points. 
    '''

    mu, sigma = model.predict(pointsConsidered, return_std=True)


   
    #add a small value to stf in order to avoid errors caused by dividing by 0
    sigma = sigma.reshape(-1, 1) + 10**(-8)
    mu_sample_opt = np.min(model.predict(XY))


    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - exploration
        Z = imp / sigma
        Z=sklearn.preprocessing.scale(Z)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)


        return ei 


def calculate_boundaries(XY, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints):

    '''
    Calculates boundries of whithin random samplePoints are generated. 

    Parameters
    ----------
    XY: numpy array (m x 2)
        Evaluated point locations.
    Z:  numpy array(m x 1)
        Values of XY points. 
    model: sklearn.gaussian_process
        A GaussianProcessRegressor fitted to points in XY and their evaluations Z.
    bounds: numpy array (2 x 2)
        Bounds of applied on XY.
    samplePoints: integer
        Number of random points for expected improvement evaluation.
    tavalist: integer
        Number of iterations before applying restricted bounds.
    suurem: integer
        Edge of the restricted bounds is suurem * np.min(Z).
    no_startingPoints: integer
        Number of random evaluations before using the expected improvement.

    Returns
    -------
    
    xmin: float
        Minimum x boundary
    xplus: float
        Maximum x boundary
    ymimn: float
         Minimum y boundary
    yplus: float
         Maximum y boundary    
       
    '''

    if len(Z) > tavalist + no_startingPoints:

        index=np.argmin(Z)
        xzero=xplus=xmin=XY[index,0]
        yzero=yplus=ymin=XY[index,1]

        while abs(model.predict(np.array([xplus,yzero]).reshape(1,2))) < (np.min(Z)*suurem) and xplus<bounds[0,1]:
            xplus=xplus+abs(bounds[0,1])/len(Z)
            
            
        while abs(model.predict(np.array([xmin,yzero]).reshape(1,2))) < (np.min(Z)*suurem)  and xmin>bounds[0,0]:
            xmin=xmin-abs(bounds[0,0])/len(Z)
            
        while abs(model.predict(np.array([xzero,yplus]).reshape(1,2))) < (np.min(Z)*suurem)  and yplus<bounds[1,1]:
            yplus=yplus+abs(bounds[1,1])/len(Z)
            
        while abs(model.predict(np.array([xzero,ymin]).reshape(1,2))) < (np.min(Z)*suurem)  and ymin>bounds[1,0]:
            ymin=ymin-abs(bounds[1,0])/len(Z)

        print(xplus, xmin, yplus, ymin)

    else:
    
        xmin=bounds[0,0]
        xplus=bounds[0,1]
        ymin=bounds[1,0]
        yplus=bounds[1,1]

    return xmin,xplus,ymin,yplus



def generate_randomArray(xplus,xmin,yplus,ymin,samplePoints):

    '''
    Generates random samplePoints within calculated boundaries. 

    Parameters
    ----------
    xmin: float
        Minimum x boundary
    xplus: float
        Maximum x boundary
    ymimn: float
         Minimum y boundary
    yplus: float
         Maximum y boundary 
    
    Returns
    -------
    randomPoints: numpy array
        (samplePoints x 2) numpy array 

    '''

    xrandomPoints=np.random.uniform(xmin, xplus, [samplePoints,1])
    yrandomPoints=np.random.uniform(ymin, yplus, [samplePoints,1])
    randomPoints=np.column_stack((xrandomPoints,yrandomPoints))

    return randomPoints


def propose_location(acquisition, XY, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints,exploration):
    

    '''
    Proposes the next sampling point by optimizing the acquisition function.

    Parameters
    ----------
    acquisition: function
        Aquisition function used to evaluate expected improvement.
    XY: numpy array (m x 2)
        Evaluated point locations.
    Z:  numpy array(m x 1)
        Values of XY points. 
    model: sklearn.gaussian_process
        A GaussianProcessRegressor fitted to points in XY and their evaluations Z.
    bounds: numpy array (2 x 2)
        Bounds of applied on XY.
    samplePoints: integer
        Number of random points for expected improvement evaluation.
    tavalist: integer
        Number of iterations before applying restricted bounds.
    suurem: integer
        Edge of the restricted bounds is suurem * np.min(Z).
    no_startingPoints: integer
        Number of random evaluations before using the expected improvement.
    exlporation: float
        Exploitation-exploration trade-off parameter.

    Returns
    -------
    min_X: numpy array (1 x 2)
       Point with optimal expected improvement. 

    '''
    dimensions=XY.shape[1]
    min_val = 10**20
    min_x=None

    xmin,xplus,ymin,yplus = calculate_boundaries(XY, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints)

    randomPoints=generate_randomArray(xmin,xplus,ymin,yplus,samplePoints)

    EI=acquisition(randomPoints.reshape(len(randomPoints), dimensions), XY, model ,exploration)
    min_index=np.argmin(EI)
    #min_val=EI[min_index,0]
    min_x=np.array([randomPoints[min_index]])
            

    return min_x