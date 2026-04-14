#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: mlRegression.py <ML_REG input>

This script extract energies, forces, and stress from ML_REG file.
Output files can plot by xmgrace.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

def read_files(input_file):
    
    if not os.path.exists(input_file):
        print(f"ERROR!\nFile: {input_file} does not exist.")
        exit(0)
    with open(input_file, 'r') as f:
        return f.readlines()

def find_section_index(lines):
    
    energy_index = force_index = stress_index = None
    for i, line in enumerate(lines):
        if 'Total energies (eV)' in line:
            energy_index = i
        if 'Forces (eV ang.^-1)' in line:
            force_index = i
        if 'Stress (kbar)' in line:
            stress_index = i
            break
    if energy_index is None:
        print("The 'Total energies (eV)' section was not found in the ML_REG file.")
        exit(0)
    if force_index is None:
        print("The 'Forces (eV ang.^-1)' section was not found in the ML_REG file.")
        exit(0)
    if stress_index is None:
        print("The 'Stress (kbar)' section was not found in the ML_REG file.")
        exit(0)
    return energy_index, force_index, stress_index

def extract_arrays(lines,
                   energy_index,
                   force_index,
                   stress_index):
    
    def parse_block(line_slice):
        return np.array([[float(x) for x in line.split()]
                         for line in line_slice if line.strip()])
 
    energy = parse_block(lines[energy_index + 2 : force_index - 1])
    force  = parse_block(lines[force_index  + 2 : stress_index - 1])
    stress = parse_block(lines[stress_index + 2 :])
    return energy, force, stress

def validate_dimensions(energy_count, force_count, stress_count):
    
    if force_count % (3 * energy_count) != 0:
        print("ERROR! Force count is not divisible by 3 * energy_count. Check ML_REG structure.")
        exit(0)
    if stress_count % (6 * energy_count) != 0:
        print("ERROR! Stress count does not match 6 * energy_count. Check ML_REG structure.")
        exit(0)

def compute_rmse(dft, mlff):
    
    return np.sqrt(np.mean((dft - mlff) ** 2))

def compute_mae(dft, mlff):
    
    return np.mean(np.abs(dft - mlff))

def compute_r2(dft, mlff):
    
    ss_res = np.sum((dft - mlff) ** 2)
    ss_tot = np.sum((dft - np.mean(dft)) ** 2)
    
    return 1 - ss_res / ss_tot

def compute_metrics(energy_per_atom, force, stress):
    
    metrics = {
        'rmse_energy': compute_rmse(energy_per_atom[:, 0], energy_per_atom[:, 1]) * 1e3,
        'rmse_force' : compute_rmse(force[:, 0], force[:, 1]),
        'rmse_stress': compute_rmse(stress[:, 0], stress[:, 1]),
        'mae_energy' : compute_mae(energy_per_atom[:, 0], energy_per_atom[:, 1]) * 1e3,
        'mae_force'  : compute_mae(force[:, 0], force[:, 1]),
        'mae_stress' : compute_mae(stress[:, 0], stress[:, 1]),
        'r2_energy'  : compute_r2(energy_per_atom[:, 0], energy_per_atom[:, 1]),
        'r2_force'   : compute_r2(force[:, 0], force[:, 1]),
        'r2_stress'  : compute_r2(stress[:, 0], stress[:, 1]),
    }
    return metrics

def write_energy(energy_per_atom, filename='Energy.dat'):
    
    with open(filename, 'w') as o:
        o.write("# Total energies per atom(eV atom^-1)\n")
        o.write("#  DFT              MLFF\n")
        for row in energy_per_atom:
            o.write(f" {row[0]:>14.6f}   {row[1]:>14.6f}\n")

def write_force(force_reshape, filename='Force.dat'):
    
    labels = ['X', 'Y', 'Z']
    with open(filename, 'w') as o:
        for i, label in enumerate(labels):
            if i > 0:
                o.write("\n")
            o.write(f"# Forces (eV ang.^-1) along {label} direction\n")
            o.write("#  DFT              MLFF\n")
            for row in force_reshape[:, i, :]:
                o.write(f" {row[0]:>14.6E}   {row[1]:>14.6E}\n")

def write_stress(stress_reshape, filename='Stress.dat'):
    
    labels = ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
    with open(filename, 'w') as o:
        for i, label in enumerate(labels):
            if i > 0:
                o.write("\n")
            o.write(f"# Stress (kbar) along {label} component\n")
            o.write("#  DFT              MLFF\n")
            for row in stress_reshape[:, i, :]:
                o.write(f" {row[0]:>14.6E}   {row[1]:>14.6E}\n")

def write_errors(metrics, filename='ERROR.dat'):
    
    lines = [
        f"RMSE of energy per atom (meV atom^-1) : {metrics['rmse_energy']:>7.3f}",
        f"RMSE of force (eV ang.^-1)            : {metrics['rmse_force']:>7.3f}",
        f"RMSE of stress (kbar)                 : {metrics['rmse_stress']:>7.3f}",
        f"MAE of energy per atom (meV atom^-1)  : {metrics['mae_energy']:>7.3f}",
        f"MAE of force (eV ang.^-1)             : {metrics['mae_force']:>7.3f}",
        f"MAE of stress (kbar)                  : {metrics['mae_stress']:>7.3f}",
        f"R-square score of energy per atom     : {metrics['r2_energy']:>7.4f}",
        f"R-square score of force               : {metrics['r2_force']:>7.4f}",
        f"R-square score of stress              : {metrics['r2_stress']:>7.4f}",
    ]
    with open(filename, 'w') as o:
        o.write('\n'.join(lines) + '\n')
    for line in lines:
        print(line)

def main():
    if '-h' in argv or len(argv) != 2:
        usage()
    
    input_file = argv[1]

    # Parse
    lines = read_files(input_file)
    energy_index, force_index, stress_index = find_section_index(lines)
    energy, force, stress = extract_arrays(lines, energy_index, force_index, stress_index)

    # Counts and derived quantities
    energy_count = len(energy)
    force_count  = len(force)
    stress_count = len(stress)
    atom_count   = force_count // (3 * energy_count)

    # Validate
    validate_dimensions(energy_count, force_count, stress_count)

    # Transform
    energy_per_atom = energy / atom_count
    force_reshape   = force.reshape((force_count // 3, 3, 2))
    stress_reshape  = stress.reshape((energy_count, 6, 2))

    # Compute metrics
    metrics = compute_metrics(energy_per_atom, force, stress)

    # Write outputs
    write_energy(energy_per_atom)
    write_force(force_reshape)
    write_stress(stress_reshape)
    write_errors(metrics)

if __name__ == '__main__':
    main()
