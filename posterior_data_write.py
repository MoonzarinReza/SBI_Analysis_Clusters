##This code generates two text files - one containing the mean and bias and the other standard deviation- for 50 
##test samples drawn from the Quijote simulations. The model is trained on the analytical simulations. 

import torch
from torch import zeros, ones
import random

import sbi
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
# read in data

import numpy as np
from collections import OrderedDict



##Reading the data corresponding to the analytcial simulations
f_r=np.genfromtxt("ff56450.txt")
theta_np1 = f_r[:,0:8]
x_np1=f_r[:,8:24]



##Removes nan values
sums=np.sum(x_np1, axis=1)
contains_no_inf = np.invert(np.isnan(sums))
theta_np1 = theta_np1[contains_no_inf]
x_np1 = x_np1[contains_no_inf]

print(np.shape(x_np1))
print(np.shape(theta_np1))




prior_len = theta_np1.shape[1]

prior = BoxUniform(-ones(prior_len)*10, ones(prior_len)*10)



theta_np = theta_np1[0:50000,:]
x_np = x_np1[0:50000,:]

print(np.shape(x_np))
print(np.shape(theta_np))

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
params=["$\Delta\Omega_m$", "$\Delta\Omega_b$", "$\Delta h$", "$\Delta n_s$", "$\Delta\sigma_8$","$\Delta M_A$", "$\Delta M_B$", "$\Delta ln \sigma$"]




##Reading test data corresponding to Quijote simulations
theta_np1 = np.genfromtxt("quijote_MOR_input.txt")

x_np1 = np.genfromtxt("quijote_MOR_summary.txt")


print(np.shape(theta_np1))
print(np.shape(x_np1))


sums=np.sum(x_np1, axis=1)
contains_no_inf = np.invert(np.isnan(sums))
theta_np1 = theta_np1[contains_no_inf]
x_np1 = x_np1[contains_no_inf]

print(np.shape(theta_np1))
print(np.shape(x_np1))


counts=50
arr=np.zeros((counts,16))
std1=np.zeros((counts,8))
for i in range(counts):
    print(i)
    inde=random.randint(0, 195000)
    
    x_o = torch.as_tensor(x_np1[inde], dtype=torch.float32)
    theta_truth=theta_np1[inde]
    arr[i,0:8]=theta_truth
    
    
    
    samples = posterior.sample((100000,), x=x_o)
    np_samples=samples.numpy()
    for j in range(8):
        arr[i,8+j]=np.mean(np_samples[:,j])
        std1[i,j]=np.std(np_samples[:,j])
        
np.savetxt('mean&bias.txt',arr) 
np.savetxt('sd.txt',std1)    
    


