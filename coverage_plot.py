##This code reads the data from 'con_int.txt' and generates the coverage plot

import numpy as np
from matplotlib import pyplot as plt

##Defining linewidths
lw=1.6
linewidths=[lw+0.5, lw,lw+0.3, lw+0.1, lw+0.5, lw, lw+0.3, lw+0.1]

##Reading data file
data=np.genfromtxt('con_int.txt')
print(np.shape(data))
data=data/200


##Confidence intervals at which the fraction of spectra recovered are calculated
con_int=np.arange(0.005,1.005,0.005)

#Defining colors
abc=['purple', 'darkolivegreen', 'lime', 'blue', 'crimson', 'cyan', 'dimgrey', 'red']

#Defining linestyles
linestyles=['dotted','solid',   'dashdot', 'dashed','dotted', 'solid',   'dashdot', 'dashed']

##Plotting
for jj in range(8):
    plt.plot(con_int,data[:,jj], color=abc[jj], linestyle=linestyles[jj], linewidth=linewidths[jj]+.5)

plt.xlabel('Confidence Interval', fontsize=14)
plt.ylabel('Fraction of Spectra Recovered', fontsize=14)
plt.xlim([0,1])
plt.ylim([0,1])
#plt.xticks([0,0.2,0.4,0.6,0.8,1.0])
plt.legend(["$\Omega_m$", "$\Omega_b$", "$h$", "$n_s$", "$\sigma_8$","$MA$", "$MB$", "$lnsigma$"], fontsize=12)
x=[0, 1]
plt.plot(x,x, linewidth=1.5, color='black')

plt.text(0.33,0.9,'Underconfident',horizontalalignment='left',fontsize=14)
plt.text(0.65,0.3,'Overconfident',horizontalalignment='left', fontsize=14)
plt.grid()


