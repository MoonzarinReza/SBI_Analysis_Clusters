##This code calculates the sum of absolute losses for all eight parameters for different number 
##of training simulations. The model is trained on the analytical simulations but tested on the quijote simulations.

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






#Reads data files
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


theta_np = theta_np1
x_np = x_np1

print(np.shape(x_np))
print(np.shape(theta_np))

# turning into tensors

theta = torch.as_tensor(theta_np, dtype=torch.float32)
theta1 = torch.as_tensor(theta_np1, dtype=torch.float32)


x = torch.as_tensor(x_np, dtype=torch.float32)
x1 = torch.as_tensor(x_np1, dtype=torch.float32)


# SNLE, SNRE, SNPE

#inferer = SNLE(prior, device="cpu")  
inde=random.randint(0, 195000)
mea=np.zeros((11,8))
mse=np.zeros((11,8))
loss=np.zeros((11,1))
for i in range(11):
    print(i)
    theta10=theta[0:5000*(i+1)]
    x10=x[0:5000*(i+1)]
    #print(np.shape(theta10))
    #print(np.shape(x10))

    inferer = SNPE(prior, density_estimator="mdn", device="cpu")

#Append training data  
    inferer = inferer.append_simulations(theta10, x10)
     
#Training the model 
    density_estimator =inferer.train()
    posterior = inferer.build_posterior(density_estimator) 


    ##Read test data 
    
    theta_np1 = np.genfromtxt("quijote_MOR_input_v2.txt")

    x_np1 = np.genfromtxt("quijote_MOR_summary_v2.txt")
    sums=np.sum(x_np1, axis=1)
    
    contains_no_inf = np.invert(np.isnan(sums))
    theta_np1 = theta_np1[contains_no_inf]
    x_np1 = x_np1[contains_no_inf]

    
    
    
    #inde=5275
    
    x_o = torch.as_tensor(x_np1[inde], dtype=torch.float32)
    theta_truth=theta_np1[inde]
    
    
    samples = posterior.sample((100000,), x=x_o)
    np_samples=samples.numpy()
    for j in range(8):
        mea[i,j]=np.mean(np_samples[:,j])
        mse[i,j]=abs(mea[i,j]-theta_truth[j])
        
for i in range(9):
    loss[i]=np.sum(mse[i,:])
        
        
a=[5, 10,15,20,25,30,35,40,45, 50, 55]

##Plotting
plt.scatter(a,loss)
plt.ylabel('Absolute Loss')
plt.xlabel('No. of training simulations (x 1000)')    

print(loss)  
np.savetxt('loss_qui_qui.txt', loss)
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    



print(inde)



