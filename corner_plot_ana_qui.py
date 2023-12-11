##This code reads parameters and observables generated from the Fast-Forward models, use those for training the model. 
##The trained model is then tested on the Quijote simulations.

#Import Modules

import torch
from torch import zeros, ones
import random
import sbi
from sbi.utils import BoxUniform
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNPE, SNLE, SNRE
from sbi.analysis import pairplot
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict



##Read the file containing the parameters [0:8] and the observables[8:24] corresponding to the FastForward Simulations for training.
f_r=np.genfromtxt("ff56450.txt")
theta_np1 = f_r[:,0:8]
x_np1=f_r[:,8:24]





prior_len = theta_np1.shape[1]

prior = BoxUniform(-ones(prior_len)*10, ones(prior_len)*10)

sums=np.sum(x_np1, axis=1)

##Removes nan values
contains_no_inf = np.invert(np.isnan(sums))
theta_np1 = theta_np1[contains_no_inf, :]
x_np1 = x_np1[contains_no_inf, :]

print(np.shape(x_np1))
print(np.shape(theta_np1))


##Generates the training data
n_tr=50000
theta_np = theta_np1[0:n_tr,:]
x_np = x_np1[0:n_tr,:]

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
 
#Training the model
density_estimator =inferer.train()
posterior = inferer.build_posterior(density_estimator)  
params=["$\Delta\Omega_m$", "$\Delta\Omega_b$", "$\Delta h$", "$\Delta n_s$", "$\Delta\sigma_8$","$\Delta M_A$", "$\Delta M_B$", "$\Delta ln \sigma$"]

params1=["$\Omega_m$", "$\Omega_b$", "$h$", "$n_s$", "$\sigma_8$","$M_A$", "$M_B$", "$ln \sigma$"]




##Reads the Quijote simulations data for testing
theta_np1 = np.genfromtxt("quijote_MOR_input_v2.txt")

x_np1 = np.genfromtxt("quijote_MOR_summary_v2.txt")


print(np.shape(theta_np1))
print(np.shape(x_np1))




##No. of test simulations
loop=5


##Indexes for the training simulations; randomly chosen; tested for other vsets
indeinde=[50000, 70000, 90000, 110000, 130000]
#indeinde=[199390, 199540, 199690, 199840, 199990]

c = ChainConsumer()
names=['Test Sample 1', 'Test Sample 2', 'Test Sample 3', 'Test Sample 4', 'Test Sample 5']

for i in range(len(indeinde)):
    
    
    inde=indeinde[i]
    theta_truth=theta_np1[inde]
    
    x_o = torch.as_tensor(x_np1[inde], dtype=torch.float32)
    
    
    samples = posterior.sample((100000,), x=x_o)
    np_samples=samples.numpy()
    
    
    myarray = np.empty(np.shape(np_samples), dtype=np.float32)
    f = np.empty(np.shape(np_samples), dtype=np.float32)
    for j in range(8):
        myarray[:,j].fill(theta_truth[j])

        ##Calculates the differences between the truths and the posterior samples
        f[:,j] = myarray[:,j]-np_samples[:,j]
        
        
    c.add_chain(f, parameters=params, name=names[i])
    

##Plotting
#colors1=['red', 'blue', 'black', 'orange', 'green']

colors1=['#800080', 'red', 'blue', 'black', 'green']



linestyle=['dotted','solid',   'dashdot', 'dashed', 'solid']
a1=1.3
a2=a1+.1
a3=a1+0.2
xxyy=np.zeros((5,2))
linewidth1=[a3,a1,a2,a2,a1]

c.configure(max_ticks=2, linestyles=linestyle, linewidths=linewidth1, colors=colors1[0:loop],diagonal_tick_labels=False, 
    tick_font_size=10, label_font_size=18,  shade=False, bar_shade=False, usetex=False, spacing=10, legend_kwargs={"loc": "upper right","fontsize": 15}) 

theta_truth1=np.zeros((8,1)) 
c.configure_truth(linestyle='dashed', color='black', linewidth=1.4)
fig=c.plotter.plot(truth=theta_truth1)   
#

#c.tight_layout()
plt.show()
print('r')



