import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time


def funktsioon(a,b,c,d):
    start_time=time.time()
    a_array=np.random.uniform(0,1,[a,2])
    b_array=np.random.uniform(0,1,[b,2])
    c_array=np.random.uniform(0,1,[c,2])
    d_array=np.random.uniform(0,1,[d,2])

    file_object = open('hyperparameetrid.txt', 'a')
    # Append 'hello' at the end of file
    file_object.write("a_array: " + str(a_array)+ '\n')
    file_object.write("b_array: " + str(b_array)+ '\n')
    file_object.write("c_array: " + str(c_array)+ '\n')
    file_object.write("d_array: " + str(d_array)+ '\n')
    # Close the file
    file_object.close()

for i in range(10):
    a=np.random.randint(0,100)
    n_iterations=np.random.randint(90,400)
    exploration=np.random.randint(0,10)
    samplePoints=500
    no_startingPoints=10
