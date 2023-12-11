##This code generates the plot of predicted (mean and errorbar) vs true value for 50 test samples 
##for all 8 cosmological parameters 

import numpy as np
from matplotlib import pyplot as plt


std1=np.genfromtxt('sd.txt')
arr=np.genfromtxt('mean&bias.txt')

parameters=["$\Omega_m$", "$\Omega_b$", "$h$", "$n_s$", "$\sigma_8$","$MA$", "$MB$", "$lnsigma$"]  

mn=1
for k in range(0,4):
    #print(k)
    ax1=plt.subplot(4,2,mn)
    plt.errorbar(arr[:,k], arr[:,k+8], yerr=std1[:,k], fmt='bo', markersize=3, ecolor='black', elinewidth=2)
    plt.plot([min(arr[:,k]), max(arr[:,k])],[min(arr[:,k]), max(arr[:,k])], color='red')
    plt.xlabel('True Value')
    plt.ylabel('Predicted')
    plt.title(parameters[k])
    ax2 = plt.subplot(4,2,mn+2, sharex=ax1)
    
    plt.scatter(arr[:,k],(arr[:,k+8]-arr[:,k])/arr[:,k],s=15, color='red', marker='x')
    plt.plot([min(arr[:,k]), max(arr[:,k])],[0,0], color='blue')
    if((mn%2)==0):
        mn=mn+3
    else:
        mn=mn+1
    
    plt.xlabel('True Value')
    plt.ylabel('Fractional Bias')
    
plt.tight_layout()
plt.figure()  

for k in range(4,8):
    #print(k)
    ax1=plt.subplot(4,2,mn)
    plt.errorbar(arr[:,k], arr[:,k+8], yerr=std1[:,k], fmt='bo', markersize=3, ecolor='black', elinewidth=2)
    plt.plot([min(arr[:,k]), max(arr[:,k])],[min(arr[:,k]), max(arr[:,k])], color='red')
    plt.xlabel('True Value')
    plt.ylabel('Predicted')
    plt.title(parameters[k])
    ax2 = plt.subplot(4,2,mn+2, sharex=ax1)
    
    plt.scatter(arr[:,k],(arr[:,k+8]-arr[:,k])/arr[:,k],s=15, color='red', marker='x')
    plt.plot([min(arr[:,k]), max(arr[:,k])],[0,0], color='blue')
    if((mn%2)==0):
        mn=mn+3
    else:
        mn=mn+1
    
    plt.xlabel('True Value')
    plt.ylabel('Fractional Bias')
plt.tight_layout()