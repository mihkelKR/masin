import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
import time

parimadX=np.array([])
parimadY=np.array([])

def kirjutaFaili(n_iterations1,exploration1,samplePoints1,no_startingPoints1):
    global parimadX
    global parimadY

    endingTime,miinimum,positsioon = optimiseerimine(n_iterations1,exploration1,samplePoints1,no_startingPoints1)

    lahedus=abs(positsioon[0]-1)+abs(positsioon[1]-1)

    if 0.05>lahedus:
        parimadX=np.append(parimadX,positsioon[0])
        parimadY=np.append(parimadY,positsioon[1])
        file_object = open('parimad.txt', 'a')
        file_object.write(str(parimadX[-1])+ " "+ str(parimadY[-1])+'\n')
        file_object.write("n_iterations: " + str(n_iterations1)+'\n')
        file_object.write("exploration: " + str(exploration1)+'\n')
        file_object.write("samplePoints: " + str(samplePoints1)+'\n')
        file_object.write("no_startingPoints: " + str(no_startingPoints1)+'\n')
        file_object.write('\n')

        file_object.close()    

    file_object = open('hyperparameetrid.txt', 'a')
    # Append 'hello' at the end of file
    file_object.write("Min: " + str(miinimum)+ "Pos: " + str(positsioon)+ "Time: " + str(endingTime)+ '\n')
    file_object.write("n_iterations: " + str(n_iterations1)+'\n')
    file_object.write("exploration: " + str(exploration1)+'\n')
    file_object.write("samplePoints: " + str(samplePoints1)+'\n')
    file_object.write("no_startingPoints: " + str(no_startingPoints1)+'\n')
    file_object.write('\n')
    # Close the file
    file_object.close()
    
     


    
def optimiseerimine(n_iterations1,exploration1,samplePoints1,no_startingPoints1):

    start_time=time.time()

    """
    -------------
    Hyperhyperparameetrid
    ------------
    """
    n_iterations=n_iterations1
    exploration=exploration1
    samplePoints=samplePoints1
    no_startingPoints=no_startingPoints1
    bounds = np.array([[-100.0, 100.0],[-100.0,100.0]])


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
        mu_sample = model.predict(XY)
       
        #reshapin sigma ja liidan vaikese arvu, et edasistes arvutustes ei oleks 0-ga jagamist
        sigma = sigma.reshape(-1, 1) + 0.0000000001
        mu_sample_opt = np.min(mu_sample)

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

    def propose_location(acquisition, XY, Z, model, boundss, samplePoints):
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

        # Find the best optimum by starting from samplePoints different random points
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

    a1=time.time()-start_time
    index=np.argmin(Z)
    a2=Z[index]
    a3=XY[index]


    return a1,a2,a3


for i in range(10000):
    
    n_iterations1=np.random.randint(50,300)
    exploration1=np.random.randint(0,100)
    samplePoints1=np.random.randint(200,1000)
    no_startingPoints1=np.random.randint(10,100)
    kirjutaFaili(n_iterations1,exploration1,samplePoints1,no_startingPoints1)
