##This code calculates the fraction of spectra recovered corresponding to different confidence intervals and writes the
## data to 'con_int.txt'. The model is trained on the analytical sims but tested on the quijote sims.


##Importing modules
import torch
from torch import zeros, ones
import sbi
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
#importing all packages used 
import numpy as np
#import matplotlib.pyplot as plt
import math
import emcee
import random
#import time

from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import interpolate
from scipy import integrate
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import interpolate
from scipy import integrate
import scipy.stats as st



##Reads the data (parameters and observables) corresponding to the Fast-Forward Simulations
f_r=np.genfromtxt("ff56450.txt")
theta_np1 = f_r[:,0:8]
x_np1=f_r[:,8:24]



prior_len = theta_np1.shape[1]

prior = BoxUniform(-ones(prior_len)*10, ones(prior_len)*10)

sums=np.sum(x_np1, axis=1)
contains_no_inf = np.invert(np.isnan(sums))
theta_np1 = theta_np1[contains_no_inf]
x_np1 = x_np1[contains_no_inf]

print(np.shape(x_np1))
print(np.shape(theta_np1))


##Generates training set
theta_np = theta_np1[0:50000,:]
x_np = x_np1[0:50000,:]

#print(np.shape(x_np))
#print(np.shape(theta_np))

# turning into tensors

theta = torch.as_tensor(theta_np, dtype=torch.float32)
theta1 = torch.as_tensor(theta_np1, dtype=torch.float32)


x = torch.as_tensor(x_np, dtype=torch.float32)
x1 = torch.as_tensor(x_np1, dtype=torch.float32)


# SNLE, SNRE, SNPE
inferer = SNPE(prior, density_estimator="mdn", device="cpu")
#inferer = SNLE(prior, device="cpu")  

# Append training data
inferer = inferer.append_simulations(theta, x)
 

##Training the model
density_estimator =inferer.train()
posterior = inferer.build_posterior(density_estimator)  



##Reads the Quijote simulations file for testing
x_np1=np.genfromtxt('quijote_MOR_summary.txt')
theta_np1=np.genfromtxt('quijote_MOR_input.txt')


print(np.shape(x_np1))
print(np.shape(theta_np1))

prior_len = theta_np1.shape[1]

prior = BoxUniform(-ones(prior_len)*10, ones(prior_len)*10)

sums=np.sum(x_np1, axis=1)
contains_no_inf = np.invert(np.isnan(sums))
theta_np1 = theta_np1[contains_no_inf, :]
x_np1 = x_np1[contains_no_inf, :]

print(np.shape(x_np1))
print(np.shape(theta_np1))






counts=200
cn=200
ci=np.zeros((cn,8))




for i in range(counts):
    print(i)
    inde1=random.randint(0, 195000)

    x_o = torch.as_tensor(x_np1[inde1], dtype=torch.float32)
    theta_truth=theta_np1[inde1]
    
    samples = posterior.sample((100000,), x=x_o)
    np_samples=samples.numpy()
    
    
    np_samples=np.sort(np_samples, axis=0)
   
      
    clow=np.zeros((cn,8)) 
    chigh=np.zeros((cn,8)) 
    xstart=50000
    xend=49999
    for ccc in range(cn):
        #print('it')
        for mmm in range(8):
            clow[ccc,mmm]=np_samples[xstart,mmm]
            chigh[ccc,mmm]=np_samples[xend,mmm]
        xstart=xstart-250
        xend=xend+250
    
    for ddd in range(cn):
        for mmm in range(8):
            if(theta_truth[mmm]>clow[ddd,mmm] and theta_truth[mmm]<chigh[ddd,mmm]):
                ci[ddd,mmm]=ci[ddd,mmm]+1

        
    

    

np.savetxt('con_int.txt', ci)
    
    
    
    
              
    

    
  









            





