##This code generates the ranks for 1000 test samples. The model is both trained and tested on the analytical simulations.

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

##Reads data
f_r=np.genfromtxt("ff56450.txt")
theta_np1 = f_r[:,0:8]
x_np1=f_r[:,8:24]


print(np.shape(x_np1))
print(np.shape(theta_np1))

prior_len = theta_np1.shape[1]

prior = BoxUniform(-ones(prior_len)*10, ones(prior_len)*10)


##Removes NaN values
sums=np.sum(x_np1, axis=1)
contains_no_inf = np.invert(np.isnan(sums))
theta_np1 = theta_np1[contains_no_inf]
x_np1 = x_np1[contains_no_inf]

print(np.shape(x_np1))
print(np.shape(theta_np1))


theta_np = theta_np1[0:50000,:]
x_np = x_np1[0:50000,:]



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








counts=1000

ranks=np.zeros((counts,8))
inde=5

##Calculates the ranks
for i in range(counts):
    print(i)
    

    x_o = torch.as_tensor(x_np1[inde], dtype=torch.float32)
    theta_truth=theta_np1[inde]
    
    
    samples = posterior.sample((1000,), x=x_o)
    np_samples=samples.numpy()
    np_samples=np.sort(np_samples)
        
    
    for mmm in range(8):
        ranks[i,mmm]=len((np.where(theta_truth[mmm]>np_samples[:,mmm]))[0])
        

np.savetxt('ranks_2.txt', ranks)            




