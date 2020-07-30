import numpy as np
import matplotlib.pyplot as plt
import bayesianfunctions3D as bf
import testfunctions as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time
from runfunction import runfunction

start_time=time.time()

"""
-------------
Hyperhyperparameetrid
------------
"""
#bounds within which the optimiser is searching for the answer
bounds = np.array([[1, 100.0],[1,100.0],[1,200.0],[1,1000000.0]])
#number of measurements on the function
n_iterations=200
#exploration coefficient
exploration=70
#how many random points the optimiser will try before deciding on the point with best expected improvement
samplePoints=200000
#how many random measurements of the function will be done before calculating expected improvement
no_startingPoints=85
#
tavalist=1000
suurem=100


#algpunktide genereerimine
XD=np.random.randint(bounds[0,0], bounds[0,1], [no_startingPoints,1])
for i in range(bounds.shape[0]-1):
    xdim=np.random.randint(bounds[(i+1),0], bounds[(i+1),1], [no_startingPoints,1])
    XD=np.column_stack((XD,xdim))

Z=runfunction(XD[0,0],XD[0,1],XD[0,2],XD[0,3])
for i in range(len(XD)-1):
    Z_next=runfunction(XD[(i+1),0],XD[(i+1),1],XD[(i+1),2],XD[(i+1),3])
    Z = np.append(Z, Z_next[0])


#Mis mudel ja Kernel
customKernel=C(1.0) * Matern()
model = GaussianProcessRegressor(kernel=customKernel)


for i in range(n_iterations):
    #print iteration
    print("New Iteration " + str(i) + "/"+str(n_iterations))

    # Update Gaussian process with existing samples
    model.fit(XD, Z)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    XD_next = bf.propose_location(bf.expected_improvement, XD, Z, model, bounds, samplePoints, tavalist,suurem, no_startingPoints,exploration)

    XD_next=np.rint(XD_next)
    
    # Obtain next sample from the objective function
    Z_next=runfunction(int(XD_next[0,0]),int(XD_next[0,1]),int(XD_next[0,2]),int(XD_next[0,3]))
    file2write=open("data.txt",'w')
    file2write.write(str(Z_next))
    file2write.write("\n")
    file2write.write(str(XD_next))
    file2write.write("\n")
    file2write.write("\n")
    file2write.close()
    # Add sample to previous samples
    
    XD = np.vstack((XD, XD_next))

    Z = np.append(Z, Z_next[0])
    

"""
------
OUTPUT
------

"""
#kui kaua l2ks aega
index=np.argmin(Z)
print(XD[index])
print(Z[index])
print(XD,Z)

file2write=open("data.txt",'w')
file2write.write(str(Z))
file2write.write("\n")
file2write.write("\n")
file2write.write(str(XD))
file2write.write("\n")
file2write.write("\n")
file2write.write(str(Z[index]))
file2write.write("\n")
file2write.write("\n")
file2write.write(str(XD[index]))
file2write.write("\n")
file2write.close()