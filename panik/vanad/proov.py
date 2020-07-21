import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time

start_time=time.time()

"""
-------------
Hyperhyperparameetrid
------------
"""
bounds = np.array([[-100.0, 100.0],[-100.0,100.0]])
n_iterations=50
exploration=100
samplePoints=500
no_startingPoints=100
tavalist=100
suurem=1000



#kirjuta funktsioon mida hakata optimiseerima
#Sisesta 2D XY array. 
def f(XY, a=1,b=100):
    x=XY[:,0]
    y=XY[:,1]
    Z=(a-x)**2+b*(y-x**2)**2
    return Z.reshape(-1,1)

#algpunktide genereerimine

XY=np.random.uniform(bounds[0,0], bounds[0,1],[no_startingPoints,2])
Z=f(XY)


#Mis mudel ja Kernel
customKernel=C(1.0) * Matern()
model = GaussianProcessRegressor(kernel=customKernel)


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
    mu_sample_opt = np.min(Z)

    with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - exploration
            Z = imp / sigma

    if sigma[0,0]>1:

        if Z<0:
            Z=Z*(-1)

        return Z

    else:
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        #ei[sigma == 0.0] = 0.0
        return ei 
    
    

#positsiooni arvutamine

def propose_location(acquisition, XY, Z, model, boundss, samplePoints, exploration):
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
    min_val = 10000000
    min_x = None


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
        randomPoints=np.concatenate((xrandomPoints,yrandomPoints),axis=1)

    else:
    
        randomPoints=np.random.uniform(boundss[0,0], boundss[0,1], size=(samplePoints, dimensions))

    for x0 in randomPoints:
        
        EI=acquisition(x0.reshape(-1, dimensions), XY, Z, model ,exploration)
        if EI < min_val:
            min_val=EI
            min_x=x0
                
    return min_x.reshape(1,2)


for i in range(n_iterations):
    #print iteration
    print("Iteration " + str(i) + "/"+str(n_iterations))

    # Update Gaussian process with existing samples
    model.fit(XY, Z)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    XY_next = propose_location(expected_improvement, XY, Z, model, bounds, samplePoints)
    
    # Obtain next noisy sample from the objective function
    Z_next = f(XY_next)
    
    # Add sample to previous samples
    XY = np.vstack((XY, XY_next))
    Z = np.vstack((Z, Z_next))
    """
    if i > tavalist:
        exploration=exploration*0.9
        suurem=suurem*0.99
    """

    """
    #Generate plot of surrogate function
    X_axis=np.linspace(bounds[0,0],bounds[0,1],1000)
    Y_axis=np.linspace(bounds[0,0],bounds[0,1],1000)
    XY=np.linspace((bounds[0,0],bounds[0,0]),(bounds[0,1],bounds[0,1]),1000)
    X,Y=np.meshgrid(X_axis,Y_axis)
    Z=model.predict(XY)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()
    """

"""
------
OUTPUT
------

"""
#kui kaua l2ks aega
print("Ending time: " + str(time.time()-start_time))

#lopp-punktid
index=np.argmin(Z)
print("Function value: "+ str(Z[index]))
print("Position: " + str(XY[index]))

#graafik proovitud punktidest
X_axis=XY[no_startingPoints:,0]
Y_axis=XY[no_startingPoints:,1]
ax = plt.axes(projection='3d')
ax.scatter(X_axis, Y_axis, Z[no_startingPoints:], cmap='viridis', linewidth=0.5)
plt.show()