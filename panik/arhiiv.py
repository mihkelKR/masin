import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import time
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

start_time=time.time()

"""
-------------
MUUUTUJAAD
------------
"""
#muutujad
n_iter=500
xi=3
n_restarts=25
mituAlgpunkti=10
bounds = np.array([[-100.0, 100.0],[-2.0,2.0]])


#kirjuta funktsioon mida lahendad
def f(X, a=1,b=100):
	x=X[:,0]
	y=X[:,1]
	funktsioon=(a-x)**2+b*(y-x**2)**2
	return funktsioon.reshape(-1,1)

#algolukorra loomine
def algolukord(mituAlgpunkti):
	algpunktid = np.random.uniform(bounds[0,0], bounds[0,1],[mituAlgpunkti,2])
	return algpunktid

X_sample=algolukord(mituAlgpunkti)
Y_sample=f(X_sample)

#Mis mudel ja Kernel
gpr = GaussianProcessRegressor()


#EI arvutamine
def expected_improvement(X, X_sample, Y_sample, gpr, xi):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    
    sigma = sigma.reshape(-1, 1) + 0.0000000001
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.min(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

#positsiooni arvutamine

def propose_location(acquisition, X_sample, Y_sample, gpr, boundss, n_restarts):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr ,xi)
    
    # Find the best optimum by starting from n_restarts different random points
    randomPoints=np.random.uniform(bounds[0,0], bounds[0,1], size=(n_restarts, dim))

    for x0 in randomPoints:
        res = minimize(min_obj, x0=x0, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(1,2)


for i in range(n_iter):
#while (np.max(f(np.linspace(bounds[:, 0], bounds[:, 1], 1000),0))-np.max(Y_sample)) > 0.0003:
    # Update Gaussian process with existing samples
    gpr.fit(X_sample, Y_sample)
    

    # Obtain next sampling point from the acquisition function (expected_improvement)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds, n_restarts)
    
    # Obtain next noisy sample from the objective function
    Y_next = f(X_next)
    
    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))
    index=np.argmin(Y_sample)
    print("Iteration " + str(i) + "/"+str(n_iter))
    print(Y_sample[index])
    print(X_sample[index])


print(np.shape(X_sample))

print("Ending time: " + str(time.time()-start_time))
