#!/usr/bin/python3

import numpy as np
import sys
import time
from numpy.fft import *

file = open(sys.argv[1],"r")

lines = file.readlines()

natoms = lines[0].split()[1]
natoms = int(natoms)
dt = float(lines[natoms+1].split()[3]) - float(lines[0].split()[3])
print('natoms:',natoms)
print('dt:',dt)

nconf = len(lines)//(natoms+1)
print("Number of configurations: ",nconf)

data = np.zeros((nconf,natoms,3))
vac = np.zeros(nconf)

for i in range(nconf): # get initial data
    for j in range(natoms):
        line  = lines[i*(natoms+1)+j+1].split()
        data[i][j] = line

norm = np.sum([np.linalg.norm(data[0][j])**2 for j in range(natoms)])

data = np.transpose(data,(1,0,2))
print(data.shape)

def xcorr(x):
    fftx = fft(x, n = 5, axis=1)
    ret = ifft(fftx * fftx.conj(), axis=1)
    ret = fftshift(ret, axes=1).real
    ret = np.sum(ret,axis=1)
    return ret

itime = 10 
tnow = time.time()
ttime = tnow

for i in range(len(data)):
    vac += xcorr(data[i])

vac /= norm

print('Done correlating. total time = {:g} seconds'.format(time.time()-ttime))

print("Writing vac.dat")
file = open("vac_new.dat","w")
for i in range(nconf):
    line = str(dt*i) + " " + str(vac[i]) + "\n"
    file.write(line)
