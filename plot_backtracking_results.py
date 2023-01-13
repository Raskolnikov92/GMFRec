import numpy as np

N = 125
lam = 1

f = open('CR_backtracking_results_(projected_POS)/N_data=%d_(anisotropic_BxBy)_lambda=%.2f.txt'%(N,lam))
# f = open('CR_backtracking_results_(projected_POS)/N_data=%d_(isotropic_BxBy)_test.txt'%N)

deltaTheta = []
deltaThetaTrue = []
for line in f:
    parts = line.split() # split line into parts
    deltaTheta.append(float(parts[0]))
    deltaThetaTrue.append(float(parts[1]))

deltaTheta = np.array(deltaTheta)
deltaThetaTrue = np.array(deltaThetaTrue)

print(len(deltaThetaTrue))

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14


#plt.title("Angle between true and reconstructed initial velocity vector")
plt.xlabel(r"$\theta$ (degrees)", fontsize = 16)
bins = 'doane'
plt.hist(deltaTheta, bins = bins, histtype='step', density = True, color = 'blue', label = 'With reconstruction', )
#plt.axvline(x=3, color = 'black')
#plt.show()

#plt.title("Angle between initial and final velocity of the true path")
plt.xlabel(r"$\theta$ (degrees)", fontsize = 16)
plt.hist(deltaThetaTrue, bins = bins, histtype='step', density = True, color = 'red', label = 'Without reconstruction')
plt.axvline(x=3, color = 'black')

plt.ylabel("Frequency (Normalised)", fontsize = 16)
#plt.title(r"$ \ell= %.d$ pc"%(3000/N**(1/3)))
plt.legend()
plt.show()
