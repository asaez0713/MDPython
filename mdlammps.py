#!/usr/bin/python3
# run MD using lammps style input

import sys
import numpy
import math
import time
import re

#import mdglobal  # file with global variables
import mdinput   # file with input routines
import mdoutput  # file with output routines
import mdbond    # file with bonding routines

# assigned global variables
global natoms       # number of atoms
global atypes       # number of atom types
global nbonds       # number of bonds
global tbonds       # number of bond types
global box          # box coordinates
global pot          # potential components
global mass         # masses of each type
global masses       # broadcast of mass type to atoms
global pos          # positions of each atom
global vel          # velocities of each atom
global acc          # acceleration of each atom
global aatype       # array of atom types
global bonds        # bonds (array with type, ibond, jbond)
global hessian      # hessian matrix
global abtype       # array of bond types
global logfile      # file to output thermodata

box = numpy.zeros(3)
pot = numpy.zeros(6)

kb = 1.38064852e-23  # Boltzmann's constant
T = 298.0            # system temp 
zeta = numpy.zeros(2) 
Q = numpy.array([1,1.1])            #to be updated - Q = 3NkT/(omega)^2 <- what are freqs?
vtherm = numpy.zeros(2)
G = numpy.zeros(2)
w = [1/(2 - 2**(1/3)),0,0]
w[2] = w[0]
w[1] = 1- 2*w[0]

#------------------------------------------------
def zero_momentum(masses,vel): #zero the linear momentum
    mom = masses*vel # get momentum
    tmom = numpy.sum(mom,axis=0)/numpy.sum(masses,axis=0) #total mom/ma
    vel -= tmom #zero out

#-------------------------------------------------
def readinit(datafile): # read lammps init data file

    global natoms, atypes, nbonds, tbonds, box

    # unpack or destructuring
    natoms, atypes, nbonds, tbonds, box[0], box[1], box[2] = mdinput.readinvals(datafile) 
    print("Natoms",natoms," Atypes",atypes," Bonds",nbonds," Btypes",tbonds)
    print("Box",box)

    # allocate arrays from data
    global mass, aatype, pos, vel, acc, masses, bonds, hessian, zeta, Q

    acc = numpy.zeros((natoms,3))

    mass, aatype, pos, vel, masses, bonds = mdinput.make_arrays(datafile,reps)

#-------------------------------------------
def readin(): # read lammps like infile

    global nsteps, dt, initfile, ithermo, idump, dumpfile, bond_style, bondcoeff
    global logfile, inmfile, inmo
    global bondcoeff, reps

    # print lines
    data, bond_style, bondcoeff, reps = mdinput.readsysvals(sys.argv[1]) # read lammps in file
    dt, initfile, bond_styles, idump, dumpfile, ithermo, logfile, inmfile, inmo, nsteps = data
    print("dt, initfile, bond_styles, idump, dumpfile, ithermo, logfile, inmfile, inmo, nsteps",data)

#-----------------------------------------------------------
def force(): # get forces from potentials
    global pot, nbonds, bonds, bondcoeff
    global masses, pos, vel, acc

    acc.fill(0) # zero out forces/acceration
    # lj
    pot[0] = 0
    # bonds
    pot[1] = mdbond.bond_force(bond_style,nbonds,bonds,bondcoeff,pos,acc,masses)
    # bend
    pot[2] = 0
    # torsion
    pot[3] = 0


#-----------------------------------------------------------
def nhchain(Q,G,dt,natoms,vtherm,zeta,ke,vel):
    
    M = len(zeta)   #chain length
    scale = 1.0
    for i in range(3):
        ts = w[i]*dt
        for j in range(1,M-2):
            G[M-j] = (Q[M-j-1]*vtherm[M-j-1]*vtherm[M-j-1] - kb*T)/Q[M-j-1]
            vtherm[M-j] += G[M-j]*ts/4.0
            vtherm[M-j-1] *= math.exp(-vtherm[M-j]*ts/8.0)
        vtherm[0] *= math.exp(-vtherm[1]*ts/8.0)
        G[0] = (ke - (3.0*natoms)*kb*T)/Q[M-2]
        vtherm[0] += G[0]*ts/4.0
        vtherm[0] *= math.exp(-vtherm[1]*ts/8.0)
        scale *= math.exp(-vtherm[0]*ts/2.0)
        ke *= math.exp(-vtherm[0]*ts)
        zeta += vtherm*ts/2.0
        vtherm[0] *= math.exp(-vtherm[1]*ts/8.0)
        G[0] = (ke - 3.0*natoms*kb*T)/Q[M-2]
        vtherm[0] += G[0]*ts/4.0
        vtherm[0] *= math.exp(-vtherm[1]*ts/8.0)
        for j in range(1,M-2):
            vtherm[j] *= math.exp(-vtherm[j+1]*ts/8.0)
            G[j] = (Q[j-1]*vtherm[j-1]*vtherm[j-1] - kb*T)/Q[j]
            vtherm[j] += G[j]*ts/4.0
            vtherm[j] *= math.exp(-vtherm[j+1]*ts/8.0)
        G[M-1] = (Q[M-2]*vtherm[M-2]*vtherm[M-2] - kb*T)/Q[M-1]
        vtherm[M-1] += G[M-1]*ts/4.0

    vel *= scale

    return ke, vel

#-----------------------------------------------------------

# read command line for input file
if (len(sys.argv) < 2):  # error check that we have an input file
    print("No input file? or wrong number of arguments")
    exit(1)
print (sys.argv)

if len(sys.argv) > 2:
    if re.search('nvt',sys.argv[2],flags=re.IGNORECASE):
        def step(): # nose-hoover thermostat
            global pos, vel, acc, dt, ke
            
            ke,vel = nhchain(Q,G,dt,natoms,vtherm,zeta,ke,vel)
            vel += acc*dt/2.0
            pos += vel*dt
            force()
            vel += acc*dt/2.0
            ke,vel = nhchain(Q,G,dt,natoms,vtherm,zeta,ke,vel)

else:
    def step(): # velocity verlet
        global pos, vel, acc, dt

        vel += acc*dt/2.0
        pos += vel*dt
        force()
        vel += acc*dt/2.0

readin() # read infile
readinit(initfile)
ke = (0.5*numpy.dot(masses.transpose()[0],numpy.array([numpy.dot(vec,vec) for vec in vel])))

# inital force and adjustments
zero_momentum(masses,vel)  # zero the momentum
force()
teng = mdoutput.write_thermo(logfile,0,natoms,masses,pos,vel,pot)

itime = 1
tnow = time.time()
ttime = tnow
tol = 1e-8
dump_vel = 5

print("Running dynamics")

eig_array = [] # empty array for the eigenvalues

for istep in range(1,nsteps+1):

    step() # take a step

    if(istep%inmo==0): # get instantaneous normal modes
        hessian = mdbond.inm(bond_style,nbonds,bonds,bondcoeff,pos,masses)

        # print(hessian)
        w,v = numpy.linalg.eig(hessian)
        # remove lowest eigegvalues (translations of entire system)
        idx = numpy.argmin(numpy.abs(w.real))
        while abs(w[idx]) < tol:
            w = numpy.delete(w,idx)
            idx = numpy.argmin(numpy.abs(w.real))
        eig_array.append(w.real) # only get real part of array - imag do to round off error is small so we throw away.

    if(istep%ithermo==0): # write out thermodynamic data
        teng = mdoutput.write_thermo(logfile,istep,natoms,masses,pos,vel,pot)

    if(istep%idump==0): # dump to xyz file so we can see this in lammps
        mdoutput.write_dump(dumpfile,istep,natoms,pos,aatype)

    if(istep%dump_vel==0):
        mdoutput.write_dump_vel("vel.dat",istep*dt,natoms,vel)

    if(itime < time.time()-tnow): # report where we are
        print('step = {}/{} = {:.4f}%, teng = {:g}, time = {:g}'.format(istep,nsteps,istep/nsteps*100,teng,time.time()-ttime))
        tnow = time.time()

print('Done dynamics! total time = {:g} seconds'.format(time.time()-ttime))
mdoutput.write_init("test.init",istep-1,natoms,atypes,nbonds,tbonds,box,mass,pos,vel,bonds,aatype)

#Create histogram!
nconf = len(eig_array)
if(nconf==0):
    print("No configurations calculated eigenvalues! thus NOT calculating historgram")
else:
    print("Creating Histogram with",len(eig_array),"configurations")
    q1, q3 = numpy.percentile(numpy.array(eig_array), [25, 75])
    iqr = q3 - q1

    fd_width = 2*iqr/(nconf**(1/3))
    fd = (numpy.amax(eig_array) - numpy.amin(eig_array))/fd_width
    fd = int(fd) + 1

    sturges = numpy.log2(nconf) + 1
    sturges = int(sturges) + 1

    bin_ct = max(fd,sturges)
    histo,histedge = numpy.histogram(numpy.array(eig_array),bins=bin_ct,density=True)
    histdat = numpy.zeros((histo.size,2))
    for i in range(histo.size):
        histdat[i][0] = (histedge[i]+histedge[i+1])/2
        histdat[i][1] = histo[i]
        #print(histo,histedge,histdat)
    head = "Histogram of eigenvalues " + sys.argv[0] + " " + str(len(eig_array))
    numpy.savetxt(inmfile,(histdat),header=head,fmt="%g")

print("Done!")
exit(0)
