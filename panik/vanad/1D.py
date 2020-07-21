import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import time

start_time=time.time()

#kirjuta funktsioon mida lahendad
def f(X, noise=0.1):
    return X**2*np.sin(5*np.pi*X/3)**6 + noise * np.random.randn(*X.shape)



#EI arvutamine
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
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

    sigma = sigma.reshape(-1, 1)
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

#positsiooni arvutamine
def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts):
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
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr ,xi)
    
    # Find the best optimum by starting from n_restarts different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')      
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)

#VARIABLES
"""
-------------
MUUUTUJAAD
------------
"""
#muutujad
n_iter=100
xi=1
n_restarts=25
mitu_algpunkti=10
noise=0.1
bounds = np.array([[-1.0, 2.0]])

#algolukorra loomine

X_sample=np.random.uniform(bounds[:,0], bounds[:,1],mitu_algpunkti).reshape(-1,1)
Y_sample=f(X_sample,noise)

#mudeli paika panemine
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

for i in range(n_iter):
#while (np.max(f(np.linspace(bounds[:, 0], bounds[:, 1], 1000),0))-np.max(Y_sample)) > 0.0003:
    # Update Gaussian process with existing samples
    gpr.fit(X_sample, Y_sample)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds, n_restarts)
    
    # Obtain next noisy sample from the objective function
    Y_next = f(X_next, noise)
    
    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))
    index=np.argmax(Y_sample)
    print("Iteration " + str(i) + "/"+str(n_iter))
    print(Y_sample[index])
    print(X_sample[index])
    


#OUTPUT
print("Ending time: " + str(time.time()-start_time))


a=np.max(Y_sample)
b=None

for i in range(len(Y_sample)):
	if Y_sample[i]==a:
		b=X_sample[i]

print(len(X_sample))
print(b)

plt.scatter(X_sample,Y_sample, s=12, c='r')
plt.scatter(np.linspace(bounds[:, 0], bounds[:, 1], 1000), f(np.linspace(bounds[:, 0], bounds[:, 1], 1000),0), s=3)
plt.show()

