##This code plots the uncertainties of $Omega_m$ vs number of training simulations for different number of variable parameters

import numpy as np
import matplotlib.pyplot as plt
iter=17
import matplotlib

a= [0] * iter
up=8
lo=2
s1=20

for i in range(iter):
    a[i]= 100*1.5**i
    #print(a[i])
a=np.array(a)
#a=np.concatenate((a,np.array([100*1.5**16])))
a1=a
#a1=np.log10(a)
fig1, ax1 = plt.subplots()


b=np.genfromtxt('unc_1_qui_qui.txt')
ax1.plot(a,b[5,:], marker='o', linewidth=1.5, markersize=4, color='blue', label='1-parameter')
ax1.fill_between(a,b[lo,:], b[up,:], color='blue', alpha=0.3)




b=np.genfromtxt('unc_3_qui_qui.txt')
ax1.plot(a,b[5,:], marker='x', linewidth=1.5, markersize=4, color='green', label='3-parameters')
ax1.fill_between(a1, b[lo, :], b[up, :], color='green', alpha=0.3)



b=np.genfromtxt('unc_5_qui_qui.txt')
ax1.plot(a,b[5,:], marker='+', linewidth=1.5, markersize=4, color='red', label='5-parameters')
ax1.fill_between(a1,b[lo,:], b[up,:], color='red', alpha=0.3)




b=np.genfromtxt('unc_8_qui_qui.txt')
c=b[5,:]
ax1.plot(a,b[5,:], marker='*', linewidth=1.5, markersize=4, color='black', label='8-parameters')
ax1.fill_between(a,b[lo,:], b[up, :], color='black', alpha=0.3)



ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks([100, 300, 1000, 3000, 10000, 30000, 100000])
ax1.tick_params(axis='both', which='major', labelsize=13)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.ylabel('Uncertainty for $\Omega_m$', fontsize=16)
plt.xlabel('Number of training simulations', fontsize=14)
