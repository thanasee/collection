#!/usr/bin/env python

from sys import argv, exit
#import subprocess

if '-h' in argv or len(argv) > 4 or len(argv) < 3:
    print ("""
Usage: vaspVibration.py <POSCAR input> <OUTCAR input> <scaling input>

This script extracts vibrational modes and writes them in XSF format.
Supports both VASP (OUTCAR) and Phonopy (band.yaml/mesh.yaml) outputs with write eigenvectors.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)

import os
import numpy as np
from ase.io import read

scale = 1
if len(argv) == 4:
    scale = float(argv[3])

poscar_file = argv[1]

if not os.path.exists(poscar_file):
    print(f"""ERROR!
File: {poscar_file} does not exist.""")
    exit(0)

structure = read(poscar_file)

input_file = argv[2]

if not os.path.exists(input_file):
    print(f"""ERROR!
File: {input_file} does not exist.""")
    exit(0)

phonopy_check = input_file.endswith(".yaml")

if phonopy_check:
    with open(input_file, 'r') as f:
        q_point = f.read().split('phonon:')[1].split('q-position:')[1:]
        q_position = []
        for i in range(len(q_point)):
            q_position.append(q_point[i].split('[')[1].split(']')[0])
        
        bands = []
        for band in q_point:
            if band:
                bands.append(band.split('frequency:')[1:])
        
        nq_point = len(bands)
        nbands = len(bands[0])
        
        qpoint_band = [['' for band in range(nbands)]
                       for qpoint in range(nq_point)]
        eigenvectors = [['' for band in range(nbands)]
                        for qpoint in range(nq_point)]
        
        for q_point in range(nq_point):
            for band in range(nbands):
                data = (bands[q_point][band].split('atom')[1:])
                eigenvectors[q_point][band] = data
                data = float(bands[q_point][band].split('eigenvector')[0])
                qpoint_band[q_point][band] = data
        natoms = len(eigenvectors[0][0]) 
        qpoint_band = np.array(qpoint_band, dtype=float)
                   
        displacements = [[[[0 for direction in range(3)]
                           for atom in range(natoms)]
                          for band in range(nbands)]
                         for qpoint in range(nq_point)]
        
        for q_point in range(nq_point) :
            for band in range(nbands) :
                for atom in range(natoms) :
                    vector = eigenvectors[q_point][band][atom]
                    for direction in range(3) :
                        data = float(vector.split('[')[direction + 1].split(',')[0])
                        displacements[q_point][band][atom][direction] = data
                        
        modes = np.array(displacements[0], dtype=float)
        total_modes = nbands

else:

    # Read input OUTCAR file
    with open(input_file, 'r') as f:
        outcar_lines = f.readlines()
    
    for line in outcar_lines:
        if 'NIONS =' in line:
            total_ions = int(line.split()[-1])
            break
    
    frequency_index = []
    for i in range(len(outcar_lines)):
        if 'Eigenvectors and eigenvalues of the dynamical matrix' in outcar_lines[i]:
            index_start = i + 2
        if '2PiTHz' in outcar_lines[i]:
            frequency_index.append(i)
    
    index_stop = frequency_index[-1] + total_ions + 2
    
    frequency = [line.split()[-8] for line in outcar_lines[index_start:index_stop] if '2PiTHz' in line]
    modes = [line.split()[3:6] for line in outcar_lines[index_start:index_stop] if ('dx' not in line) and ('2PiTHz' not in line)]
    modes = [mode for mode in modes if len(mode) > 0]
    
    frequency = np.array(frequency, dtype=float)
    modes = np.array(modes, dtype=float).reshape((-1, total_ions, 3))
    
    modes /= np.sqrt(structure.get_masses()[None, :, None])
    
    total_modes = len(frequency)

for j in range(total_modes):
    vector = np.array(modes[j], dtype=float) * scale
    if vector.shape != structure.positions.shape:
        print("ERROR!! Shape mismatch between eigenvectors and atomic positions!")
        exit()
    positions_vector = np.hstack((structure.positions, vector))
    total_atoms = positions_vector.shape[0]
    symbols = structure.get_chemical_symbols()
    i = j + 1 if phonopy_check else total_modes - j
    
    output_name = f"mode_{i:d}.xsf"
    
    with open(output_name, 'w') as o:
        o.write("CRYSTAL\n")
        o.write("PRIMVEC\n")
        o.write("\n".join([' '.join([f'{a:20.16f}' for a in lattice])
                           for lattice in structure.cell]))
        o.write("\nPRIMCOORD\n")
        o.write(f"{total_atoms:3d} 1\n")
        o.write("\n".join([f'{symbols[k]:3s}' + ' '.join([f'{a:20.16f}' for a in positions_vector[k]])
                           for k in range(total_atoms)]))

