#!/usr/bin/python3

import numpy
import sys
import time

file = open("vel.dat","r")

lines = file.readlines()

nwindow = 1024
natoms = lines[0].split()[1]
natoms = int(natoms)
dt = float(lines[natoms+1].split()[3]) - float(lines[0].split()[3])
print(natoms,nwindow,dt)

nconf = len(lines)//(natoms+1)
print("Number of configurations: ",nconf)

data = numpy.zeros((nwindow,natoms,3))
vac = numpy.zeros(nwindow)

for i in range(nwindow): # get initial data
    for j in range(natoms):
        line  = lines[i*(natoms+1)+j+1].split()
        data[i][j] = line

itime = 1
tnow = time.time()
ttime = tnow

print("Correlating")
for i in range(nconf-nwindow):
    # Correlate
    for k in range(nwindow):
        for j in range(natoms):
            vac[k] += numpy.dot(data[k][j],data[0][j])

    # get next data window
    for j in range(nwindow-1): # move array down
        data[j] = data[j+1]
    for j in range(natoms): # read data into end of array
        data[-1][j] = lines[i*(natoms+1)+j+1].split()

    if(itime < time.time()-tnow): # report where we are
        print('conf = {}/{} = {:.4f}%, time = {:g}'.format(i,nconf,i/nconf*100,time.time()-ttime))
        tnow = time.time()

print('Done correlating. total time = {:g} seonds'.format(time.time()-ttime))

vac /= nwindow

print("Writing vac.dat")
file = open("vac.dat","w")
for i in range(nwindow):
    line = str(dt*i) + " " + str(vac[i]) + "\n"
    file.write(line)
