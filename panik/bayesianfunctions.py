import numpy as np
from scipy.stats import norm
import sklearn



#Expected improvementi arvutamine
def expected_improvement(sample, XY, Z, model, exploration):
    
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
    sigma = sigma.reshape(-1, 1) + 0.0000000001
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

def propose_location(acquisition, XY, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints,exploration):
    
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
    dimensions = XY.shape[1]
    min_val = 1000000000000000000000000
    min_x=None

    if len(Z) > tavalist+no_startingPoints:

        index=np.argmin(Z)
        xzero=xplus=xmin=XY[index,0]
        yzero=yplus=ymin=XY[index,1]

        while abs(model.predict(np.array([xplus,yzero]).reshape(1,2))) < (np.min(Z)*suurem) and abs(xplus)<100:
            xplus=xplus+bounds[0,1]/len(Z)/np.log(len(Z))
            
        while abs(model.predict(np.array([xmin,yzero]).reshape(1,2))) < (np.min(Z)*suurem)  and abs(xmin)<100:
            xmin=xmin-bounds[0,1]/len(Z)/np.log(len(Z))
            
        while abs(model.predict(np.array([xzero,yplus]).reshape(1,2))) < (np.min(Z)*suurem)  and abs(yplus)<100:
            yplus=yplus+bounds[0,1]/len(Z)/np.log(len(Z))
            
        while abs(model.predict(np.array([xzero,ymin]).reshape(1,2))) < (np.min(Z)*suurem)  and abs(ymin)<100:
            ymin=ymin-bounds[0,1]/len(Z)/np.log(len(Z))
            
        print(xplus,xmin,yplus,ymin)


        # Find the best optimum by starting from samplePoints different random points
        xrandomPoints=np.random.uniform(xmin, xplus, [samplePoints,1])
        yrandomPoints=np.random.uniform(ymin, yplus, [samplePoints,1])
        randomPoints=np.column_stack((xrandomPoints,yrandomPoints))

    else:
    
        randomPoints=np.random.uniform(bounds[0,0], bounds[0,1], size=(samplePoints, dimensions))
        
        
    EI=acquisition(randomPoints.reshape(len(randomPoints), dimensions), XY, Z, model ,exploration)
    

    min_index=np.argmin(EI)
    min_val=EI[min_index,0]
    min_x=np.array([randomPoints[min_index]])
            

    return min_x