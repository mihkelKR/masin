import numpy as np
import matplotlib.pyplot as plt
import bayesianfunctions as bf
import testfunctions as fu
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time

start_time=time.time()

"""
-------------
Hyperhyperparameetrid
------------
"""
#Bounds within which the optimiser is searching for the answer
bounds = np.array([[-100.0, 100.0],[-100.0,100.0]])
#Number of iterations
n_iterations=400
#Exploitation-exploration trade-off parameter
exploration=100
#Number of random points considered for expected improvement
samplePoints=10000
#Number of random measurements done before applying Bayesian optimisation
no_startingPoints=100
#Number of evaluation before applying restricted boundries
tavalist=200
suurem=5000

#Mis mudel ja Kernel
customKernel=C(1.0) * Matern()
model = GaussianProcessRegressor(kernel=customKernel)

a=np.random.randint(8,9)
b=np.random.randint(80,100)


#algpunktide genereerimine
XY,Z=bf.generate_startingArrays(bounds,no_startingPoints, a, b)



for i in range(n_iterations):
    #print iteration
    print("Iteration " + str(i) + "/"+str(n_iterations))

    # Update Gaussian process with existing samples
    model.fit(XY, Z)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    XY_next = bf.propose_location(bf.expected_improvement, XY, Z, model, bounds, samplePoints, tavalist,suurem, no_startingPoints,exploration)
        
    # Obtain next sample from the objective function
    Z_next = fu.rosenbrock(XY_next,a,b)
    print(Z_next)
        
    #Changes kernel for very small values

    if Z_next>10:
        model = GaussianProcessRegressor(kernel=customKernel)
    else:
        model= GaussianProcessRegressor()


    
    # Add sample to previous samples
    XY = np.vstack((XY, XY_next))
    Z = np.vstack((Z, Z_next))


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