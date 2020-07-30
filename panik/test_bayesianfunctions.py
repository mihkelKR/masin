import numpy as np 
from numpy import random 
import pytest
import bayesianfunctions2 as bf 
import testfunctions as fu
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF

'''
Hyperparameters
'''

#Bounds within which the optimiser is searching for the answer
bounds = np.array([[-100.0, 100.0],[-100.0,100.0]])
#Number of iterations
n_iterations=200
#Exploitation-exploration trade-off parameter
exploration=10
#Number of random points considered for expected improvement
samplePoints=1000
#Number of random measurements done before applying Bayesian optimisation
no_startingPoints=100
#Number of evaluation before applying restricted boundries
tavalist=100
suurem=100000
a,b=1,100

#Mis mudel ja Kernel
customKernel=C(1.0) * Matern()
model = GaussianProcessRegressor(kernel=customKernel)




def test_startingArrays_shape():

    XY,Z= bf.generate_startingArrays(bounds, no_startingPoints,a,b)
    assert np.shape(XY) == (100, 2)
    assert np.shape(Z) == (100, 1)




def test_startingArrays_value():


    XY,Z= bf.generate_startingArrays(bounds, no_startingPoints,a,b)
    assert Z[54]==fu.rosenbrock(np.array([XY[54]]),a,b)
    assert Z[0]==fu.rosenbrock(np.array([XY[0]]),a,b)
    



def test_expected_improvement_shape():

    XY,Z= bf.generate_startingArrays(bounds, no_startingPoints,a,b)
    randomPoints=np.random.uniform(1,100,[samplePoints,2])
    ei=bf.expected_improvement(randomPoints.reshape(1000,2), XY, model ,exploration)

    assert np.shape(randomPoints) == (1000,2)
    assert np.shape(ei) == (1000,1000)



def test_expected_improvement_values():

    XY,Z= bf.generate_startingArrays(bounds, no_startingPoints,a,b)
    randomPoints=np.random.uniform(1,100,[samplePoints,2])
    ei=bf.expected_improvement(randomPoints.reshape(1000,2), XY, model ,exploration)

    assert np.max(ei)<5
    assert np.min(ei)>(-5)


def test_boundaries_type():


    XY,Z= bf.generate_startingArrays(bounds, no_startingPoints,a,b)
    x1,x2,x3,x4=bf.calculate_boundaries(XY, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints)

    assert type(x1) == np.float64
    assert type(x2) == np.float64
    assert type(x3) == np.float64
    assert type(x4) == np.float64



def test_boundaries_values():

    XY,Z= bf.generate_startingArrays(bounds, no_startingPoints,a,b)
    x1,x2,x3,x4=bf.calculate_boundaries(XY, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints)

    assert x1+1 > bounds[0,0]
    assert x1-1 < bounds[0,1]
    assert x1+1 > bounds[1,0]
    assert x1-1 < bounds[0,1]


def test_randomArray():

    XY,Z= bf.generate_startingArrays(bounds, no_startingPoints,a,b)
    x1,x2,x3,x4=bf.calculate_boundaries(XY, Z, model, bounds, samplePoints, tavalist, suurem, no_startingPoints)
    randomPoints=bf.generate_randomArray(x1,x2,x3,x3,samplePoints)

    assert np.shape(randomPoints) == (samplePoints, 2)



def test_propose_location():

    testPoints=np.array([[ 84.08189903 , 53.88447462],
        [ 61.5551066 , -64.29825386],
        [ 75.47769429,  54.95134825],
        [ 84.25922798,  22.95005061],
        [ 18.76111511 , 89.09070722],
        [  8.15127747 ,-92.99402481],
        [-25.61173004, -23.33976507],
        [-14.86102206 ,  7.27895495],
        [ 33.71775302 , 12.70205813],
        [ 35.24789801,  -6.93496912],
        [-92.25707023,  71.67249785],
        [ 62.3196444,  -20.87827044],
        [-88.05612662,  95.53736364],
        [-12.62324252,  -3.87104883],
        [-20.43942307, -13.84960117],
        [-55.42548441 , -1.11338184],
        [ 82.31052265 , 12.93048874],
        [-15.58920443 , 30.86397927],
        [ 53.56635882 ,-14.9354854 ],
        [-44.84736361 ,-92.89749141]])

    XY=np.array([
    [ 19.47821846 ,-65.41573353],
    [-80.40103345,  96.0157902 ],
    [-99.87463404,   6.40045169],
    [ 29.58690441,  58.25782482],
    [  5.42079326, -58.55158147],
    [ 87.26074415,  25.22074408],
    [-53.04876964, -89.18608338],
    [-60.02006349, -42.98304316],
    [ 50.35854239, -85.72030194],
    [ 22.65360388, -70.46687959],
    [ 65.01411751,  91.40091904],
    [ 22.75455218, -51.27293991],
    [-59.68363255,  59.02380332],
    [ 84.17530091, -78.57883255],
    [ 12.56623088,  35.07843229],
    [-26.47975399,   5.24286714],
    [ -8.62250079, -21.40357284],
    [-64.81643348 , 79.85545639],
    [-51.20694985,  73.96611861],
    [ 55.86842228 ,-43.79438037],
    [ 26.03741457 , 62.63048365],
    [  2.1171202  ,-31.72578485],
    [ -6.80284297 , 25.35905382],
    [ -7.54033452 , -9.0840318 ],
    [ 11.76948866 , 46.5388806 ],
    [ -2.1750983  ,  5.68684575],
    [  6.75726675 , 42.07161477],
    [ -0.70873281  ,10.65175311],
    [ -7.18789048 , 17.28889597]])

    Z=np.array([[1.30590425e+07],[2.67666254e+09],[6.55855227e+09],[4.40684909e+07],[5.10370085e+05],[3.80135634e+09],[5.56349809e+08],[8.77070309e+08],[4.53641469e+08],[2.24831724e+07],[1.12872342e+09],[2.13716728e+07],[8.09942918e+08],[3.38737394e+09],[9.95814757e+05],[3.19665600e+07],[6.05349998e+05],[1.12103060e+09],[4.28557876e+08],[6.61170567e+08],[2.49889104e+07],[8.65510316e+04],[2.90741280e+04],[2.87190827e+05],[5.58427968e+05],[1.44476101e+02],[8.50218276e+02],[6.85817392e+03],[7.81980791e+04]])  

    model.fit(XY, Z)

    EI=bf.expected_improvement(testPoints,XY,model,exploration)
    measuredEI=np.array([[ 8.80135006e+08],[ 6.68087696e+08],[ 7.49639617e+08],[ 3.69537226e+09],[ 3.99061124e+07],[ 2.96901189e+06],[ 4.62467379e+07],[-7.01390727e+06],[ 1.21828286e+08],[ 1.00388328e+08],[ 7.32161089e+08],[ 2.71093649e+08],[ 2.16041447e+09],[-3.33451476e+06],[ 1.47934999e+07],[ 5.39245334e+08],[ 2.29868990e+09],[ 1.07799278e+07],[ 2.18349647e+08],[ 1.82712005e+08]])

    assert EI[1] == pytest.approx(measuredEI[1])
    assert EI[-1]==pytest.approx(measuredEI[-1])