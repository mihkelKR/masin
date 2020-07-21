import numpy as np
import matplotlib.pyplot as plt
import bayesianfunctions2 as bf
import testfunctions as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time

start_time=time.time()

"""
-------------
Hyperhyperparameetrid
------------
"""
#bounds within which the optimiser is searching for the answer
bounds = np.array([[-100.0, 100.0],[-100.0,100.0]])
#number of measurements on the function
n_iterations=400
#exploration coefficient
exploration=70
#how many random points the optimiser will try before deciding on the point with best expected improvement
samplePoints=500
#how many random measurements of the function will be done before calculating expected improvement
no_startingPoints=100
#
tavalist=100
suurem=1000


#algpunktide genereerimine

X=np.random.uniform(bounds[0,0], bounds[0,1], [no_startingPoints,1])
Y=np.random.uniform(bounds[1,0], bounds[1,1], [no_startingPoints,1])
XY=np.column_stack((X,Y))
Z=tf.rosenbrock(XY,1,100)


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
    Z_next = tf.rosenbrock(XY_next,1,100)
    print(Z_next)

    if Z_next>100:
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