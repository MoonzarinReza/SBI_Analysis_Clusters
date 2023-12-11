##This code reads the ranks from 'ranks_2.txt' and plots a histogram of ranks for all 8 parameters


##Imports modules
from matplotlib import pyplot as plt
import numpy as np

parameters=[" $\Omega_m$", " $\Omega_b$", " $h$", " $n_s$", " $\sigma_8$", " $M_A$", " $ M_B$", " $lnsigma$"]


##Reads the data for the ranks
x=np.genfromtxt('ranks_qui.txt')
kk=0
aa=2
bb=4
for ii in range(2):
    for jj in range(4):
        if(ii==0):
             plt.subplot(2,4,jj+1)
             n, bins, patches =plt.hist(x[:,kk], 50, density=False, color=['blue'], histtype='step')
             #print(len(np.where(n>4)[0]))
             #print(parameters[kk])
             
             x5=np.arange(0,1001,1)
             y1=np.full((len(x5),1),4)
             y2=np.full((len(x5),1),0)
             #plt.plot(x5,y1, linewidth=2, color='red')
             #plt.plot(x5,y2, linewidth=2, color='red')
             plt.ylim(-1,100)
            
            
            

             
             plt.title('Rank for' + parameters[kk])
             kk=kk+1
             plt.tight_layout()
             
        else:
            plt.subplot(2,4, jj+5)
            n,bins, patches=plt.hist(x[:,kk], 50, density=False, color=['blue'], histtype='step')
            print(parameters[kk])
            
            print(len(np.where(n>4)[0]))
            x5=np.arange(0,1001,1)
            y1=np.full((len(x5),1),4)
            y2=np.full((len(x5),1),0)
            #plt.plot(x5,y1, linewidth=2, color='red')
            #plt.plot(x5,y2, linewidth=2, color='red')
            plt.ylim(-1,100)
            
            
            
            plt.title('Rank for' + parameters[kk])
            kk=kk+1
            plt.tight_layout()
            
          
plt.show()
    
        
       
  
        
         
    
    
    


    
    






