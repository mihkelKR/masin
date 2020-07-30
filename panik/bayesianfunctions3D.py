import numpy as np
from scipy.stats import norm
import sklearn



#Expected improvementi arvutamine
def expected_improvement(sample, XD, Z, model, exploration):
    
    '''
    Computes the EI at points X based on existing samples XY
    and Z using a Gaussian process surrogate model.
    
    Args:
        sample: Points at which EI shall be computed (m x d).
        XY: Sample locations (n x d).
        Z: Sample values (n x 1).
        model: A GaussianProcessRegressor fitted to samples.
        exlporation: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points "sample".
    '''
    
    mu, sigma = model.predict(sample, return_std=True)
   
    #reshapin sigma ja liidan vaikese arvu, et edasistes arvutustes ei oleks 0-ga jagamist
    sigma = sigma + 0.0000000001
    #print(np.column_stack((mu,sigma)))
    mu_sample_opt = np.min(Z)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - exploration
        Z = imp / sigma
        
        Z=sklearn.preprocessing.scale(Z)
        
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        #ei[sigma == 0.0] = 0.0
            

        return ei 



#positsiooni arvutamine

def propose_location(acquisition, XD, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints,exploration):
    
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        XY: Sample locations (n x d).
        Z: Sample values (n x 1).
        model: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dimensions = XD.shape[1]

    min_val = 1000000000000000000000000
    min_x=None

    if len(Z) > tavalist+no_startingPoints:

        index=np.argmin(Z)
        minPoint=XD[index].reshape(1,4)
        
        maxBound=minBound=minPoint[0,0]

        while model.predict(minPoint)< (np.min(Z)*suurem) and maxBound<bounds[0,1]:
            maxBound=maxBound+abs(bounds[0,1])/len(Z)
            minPoint[0,0]=maxBound
            

        minPoint=XD[index].reshape(1,4)
        while model.predict(minPoint) < (np.min(Z)*suurem) and minBound>bounds[0,0]:
            minBound=minBound-abs(bounds[0,0])/len(Z)
            minPoint[0,0]=minBound

        randomPoints=np.random.uniform(minBound, maxBound, [samplePoints,1])
        print(minBound,maxBound)

        for i in range(dimensions-1):
            minPoint=XD[index].reshape(1,4)
            maxBound=minBound=minPoint[0,i]

            while model.predict(minPoint) < (np.min(Z)*suurem) and maxBound<bounds[i,1]:
                maxBound=maxBound+abs(bounds[i,1])/len(Z)
                minPoint[0,i]=maxBound
                

            minPoint=XD[index].reshape(1,4)
            while model.predict(minPoint) < (np.min(Z)*suurem) and minBound>bounds[i,0]:
                minBound=minBound-abs(bounds[i,0])/len(Z)
                minPoint[0,i]=minBound

            xirandomPoints=np.random.uniform(minBound, maxBound, [samplePoints,1])
            randomPoints=np.column_stack((randomPoints,xirandomPoints))
            print(minBound,maxBound)

    else:
        randomPoints=np.random.uniform(bounds[0,0], bounds[0,1], [samplePoints,1])
        for i in range(dimensions-1):
            xirandomPoints=np.random.uniform(bounds[(i+1),0], bounds[(i+1),1], [samplePoints,1])
            randomPoints=np.column_stack((randomPoints,xirandomPoints))

       
    EI=acquisition(randomPoints.reshape(len(randomPoints), dimensions), XD, Z, model ,exploration)

    

    min_index=np.argmin(EI)
    min_val=EI[min_index]
    min_x=np.array([randomPoints[min_index]])

    return min_x