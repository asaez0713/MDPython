# import routines for MDPython

import re
import numpy as np

global natoms
global atypes
global nbonds
global tbonds
global box

box = np.zeros(3)

re_dict_data = {
        'natoms': re.compile(r'(?P<natoms>\d+) atoms\n'),
        'atypes': re.compile(r'(?P<atypes>\d+) atom types\n'),
        'nbonds': re.compile(r'(?P<nbonds>\d+) bonds\n'),
        'tbonds': re.compile(r'(?P<tbonds>\d+) bond types\n'),
        'box_x': re.compile(r'(?P<box_x>[\d-]+ [\d-]+) xlo xhi\n'),
        'box_y': re.compile(r'(?P<box_y>[\d-]+ [\d-]+) ylo yhi\n'),
        'box_z': re.compile(r'(?P<box_z>[\d-]+ [\d-]+) zlo zhi\n')
        }

def parse_line(line,d):
    for key, rx in d.items():
        match = rx.search(line)
        if match:
            return key, match

    return None, None

def readinvals(datafile):
    with open(datafile,'r') as f:
        data = []
        line = f.readline()
        while line:
            key, match = parse_line(line,re_dict_data)
            for item in re_dict_data:
                if key == item:
                    val = match.group(item)
                    val = val.split()
                    if len(val) > 1:
                        data.append(float(val[1])-float(val[0]))
                    else:
                        data.append(int(val[0]))
            line = f.readline()
    return data

re_dict_arrays = {
        'masses': re.compile(r'Masses\n'),
        'atoms': re.compile(r'Atoms'),
        'vels': re.compile(r'Velocities\n'),
        'bonds': re.compile(r'Bonds\n')
        }

def make_arrays(datafile,reps):
    global natoms, atypes, nbonds, tbonds, box
    
    natoms, atypes, nbonds, tbonds, box[0], box[1], box[2] = readinvals(datafile) 
    
    with open(datafile,'r') as f:
        mass = []
        aatype = []
        pos = []
        vel = []
        masses = []
        bonds = []
        
        line = f.readline()
        while line:
            key, match = parse_line(line,re_dict_arrays)
            count = 1
            if key == 'masses':
                line = f.readline()
                while len(line.split()) == 0:
                    line = f.readline()
                for i in range(atypes):
                    words = line.split()
                    if int(words[0]) != count:
                        print('Error while assigning masses')
                        print('Expecting',count,'got',words[0])
                        exit(1)
                    m = float(words[1])
                    mass.append(m)
                    count += 1
                    line = f.readline()
                print('Assigned masses')
            if key == 'atoms':
                line = f.readline()
                while len(line.split()) == 0:
                    line = f.readline()
                for i in range(natoms):
                    words = line.split()
                    if int(words[0]) != count:
                        print('Error while assigning atoms')
                        print('Expecting',count,'got',line.split()[0])
                        exit(1)
                    atype = int(words[1])
                    aatype.append(atype)
                    pos.append([float(words[j + 4]) for j in range(3)])
                    masses.append([mass[atype - 1] for j in range(3)])
                    count += 1
                    line = f.readline()
                print('Assigned atom types and positions')
            if key == 'vels':
                line = f.readline()
                while len(line.split()) == 0:
                    line = f.readline()
                for i in range(natoms):
                    words = line.split()
                    if int(words[0]) != count:
                        print('Error while assigning velocities')
                        print('Expecting',count,'got',line.split()[0])
                        exit(1)
                    vel.append([float(words[j + 1]) for j in range(3)])
                    count += 1
                    line = f.readline()
                print('Assigned velocities')
            if key == 'bonds':
                line = f.readline()
                while len(line.split()) == 0:
                    line = f.readline()
                for i in range(nbonds):
                    words = line.split()
                    if int(words[0]) != count:
                        print('Error while assigning velocities')
                        print('Expecting',count,'got',line.split()[0])
                        exit(1)
                    bonds.append([int(words[j + 1])-1 for j in range(3)])
                    count += 1
                    line = f.readline()
                print('Assigned bonds')
            line = f.readline()
    
    if len(vel) == 0:
        vel = np.zeros((natoms,3))
        print('Velocities will start at 0')
    if len(bonds) == 0:
        print('No bonds found')

    pos_copy = pos[:]
    vel_copy = vel[:]
    bonds_copy = bonds[:]
    count = 1 

    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                if i == j == k == 0:
                    continue
                offset = [i*box[0],j*box[1],k*box[2]]
                for vec in pos_copy:
                    temp = [sum(x) for x in zip(vec,offset)]
                    pos.append(temp)
                bond_offset = [0,count*len(pos_copy),count*len(pos_copy)]
                for bond in bonds_copy:
                    temp = [sum(x) for x in zip(bond,bond_offset)]
                    bonds.append(temp)
                count += 1

    fact = np.prod(reps)
    natoms *= fact
    nbonds *= fact
    box = [np.prod(x) for x in zip(box,reps)]

    return np.array(mass), np.array(aatype), np.array(pos), np.array(vel), np.array(masses), np.array(bonds)

re_dict_sysvals = {
        'nsteps': re.compile(r'run (?P<nsteps>\d+)'),
        'dt': re.compile(r'timestep (?P<dt>[\d.]+)'),
        'initfile': re.compile(r'read_data (?P<initfile>[ a-z.]+)'),
        'ithermo': re.compile(r'thermo (?P<ithermo>\d+)'),
        'dump': re.compile(r'dump traj all xyz (?P<dump>\d+ [a-z.]+)'),
        'bond_style': re.compile(r'bond_style (?P<bond_style>[a-z]+)'),
        'logfile': re.compile(r'log (?P<logfile>[a-z.]+)'),
        'inm': re.compile(r'inm (?P<inm>[a-z.]+ \d+)'),
        'reps': re.compile(r'replicate (?P<reps>\d+ \d+ \d+)')
        }

def readsysvals(infile):
    with open(infile,'r') as f:
        data = []
        bondcoeff = []
        reps = [1,1,1]
        line = f.readline()
        while line:
            key, match = parse_line(line,re_dict_sysvals)
            if key in ['nsteps','ithermo']:
                data.append(int(match.group(key)))
            if key == 'dt':
                data.append(float(match.group(key)))
            if key == 'initfile':
                initfile = match.group(key).strip(' ')
                data.append(initfile)
                natoms, atypes, nbonds, tbonds, box[0], box[1], box[2] = readinvals(initfile) 
            if key in ['logfile','bond_style']:
                data.append(match.group(key).strip(' '))
            if key == 'dump':
                val = match.group(key).split()
                data.append(int(val[0]))
                data.append(val[1])
            if key == 'inm':
                val = match.group(key).split()
                data.append(val[0])
                data.append(int(val[1]))
            if key == 'bond_style':
                bond_styles = match.group(key)
                if re.search('harmonic',bond_styles,flags=re.IGNORECASE):
                    bond_style = 0
                    print('Reading in harmonic bond coefficients for',tbonds,'types')
                    for i in range(tbonds):
                        line = f.readline()
                        bondcoeff.append([float(line.split()[j+2]) for j in range(2)])
                elif re.search('morse',bond_styles,flags=re.IGNORECASE):
                    bond_style = 1
                    print('Reading in morse bond coefficients for',tbonds,'types')
                    for i in range(tbonds):
                        line = f.readline()
                        bondcoeff.append([float(line.split()[j+2]) for j in range(3)])
                else:
                    print('Unrecognized bond type')
            if key == 'reps':
                vals = match.group(key).split()
                reps = [int(num) for num in vals]
            line = f.readline()
    
    bondcoeff = np.array(bondcoeff)

    return data, bond_style, bondcoeff, reps
