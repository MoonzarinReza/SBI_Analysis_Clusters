##This code calculates the uncertainties in $Omega_m$ for different number of training simulations for different number of variable parameters

import torch
from torch import zeros, ones

import sbi
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import pandas as pd

#inde=13757

#a=['bs-', 'rs-', 'ks-']
#loo=1
loo=10
iter=17

unc_om=np.zeros((loo,iter))
a= [0] * iter

for i in range(iter):
    a[i]= 100*1.474**i
    print(a[i])

##Reads data files
theta_np = (np.genfromtxt("quijote_MOR_input_v2.txt"))
print(np.shape(theta_np))

#print(theta_np[0:1000:100])
x_np = np.genfromtxt("quijote_MOR_summary_v2.txt")



##Removes NaN values
sums=np.sum(x_np, axis=1)
contains_no_inf = np.invert(np.isnan(sums))
theta_np = theta_np[contains_no_inf]
x_np = x_np[contains_no_inf]
print(np.shape(theta_np))
print(np.shape(x_np))
shapes_theta = np.shape(theta_np)



inde=random.randint(0, 195000)
#prior_len = shapes_theta[1]
prior_len=1
prior = BoxUniform(-ones(prior_len)*10, ones(prior_len)*10)
print(np.shape(prior))
params = ["$\Omega_m$"]
#print(np.shape(theta_np))

for i in range(loo):
    
    
    for jj in np.arange(iter):
        print(i)
        print(jj)
        #k=a[jj]
        #print(k)
        theta_np1 = theta_np[0:int(a[jj])]
        #print(np.shape(theta_np1))
        x_np1 = x_np[0:int(a[jj])] 
        theta = torch.as_tensor(theta_np1, dtype=torch.float32)
        x = torch.as_tensor(x_np1, dtype=torch.float32)
        #print(np.shape(theta_np1))
        #print(np.shape(x_np1))


    # Create inference object: choose method and estimator
        inferer_analytical = SNPE(prior, density_estimator="mdn", device="cpu")  # SNLE, SNRE
    # Append training data
        inferer_analytical = inferer_analytical.append_simulations(theta, x)

    # Train
        density_estimator_analytical = inferer_analytical.train()  
        posterior_analytical = inferer_analytical.build_posterior(density_estimator_analytical)  # Posterior sampling settings.

        theta_np1 = np.genfromtxt("quijote_MOR_input_v2.txt")

        x_np1 = np.genfromtxt("quijote_MOR_summary_v2.txt")
        sums=np.sum(x_np1, axis=1)
        
        contains_no_inf = np.invert(np.isnan(sums))
        theta_np1 = theta_np1[contains_no_inf]
        x_np1 = x_np1[contains_no_inf]

    
        theta_o = torch.as_tensor(theta_np1[inde], dtype=torch.float32)
    
        x_o = torch.as_tensor(x_np1[inde], dtype=torch.float32)
    
        samples = posterior_analytical.sample((100000,), x=x_o)
    
        np_samples=samples.numpy()
   
        
        unc_om[i,jj]=np.std(np_samples[:,0])
        
        



print(np.shape(unc_om))
np.savetxt('unc_8_qui_qui.txt', unc_om)












