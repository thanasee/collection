#!/usr/bin/env python

from sys import argv, exit
import os, re
import numpy as np
import h5py as h5

def usage():
    
    text = """
Usage: convergePhono3py.py

This script obtain lattice thermal conductivity depends on q-mesh form HDF5 files

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

def find_kappa_files():
    
    def mesh_number(filename):
        match = re.search(r"kappa-m(\d+)\.hdf5", filename)
        return int(match.group(1)) if match else float('inf')
 
    files = [f for f in os.listdir() if f.startswith("kappa-m") and f.endswith(".hdf5")]
    if not files:
        print("ERROR! kappa hdf5 files are not found.")
        exit(0)
    return sorted(files, key=mesh_number)

def read_HDF5(filepath):
    
    def load(f, key):
        if key not in f:
            return None
        arr = np.array(f[key])
        # Suppress numerical noise
        return np.where(arr < 1e-12, 0.0, arr)

    with h5.File(filepath, 'r') as f:
        data = {'mesh': np.array(f["mesh"]) if 'mesh' in f else None,
                'temperature': np.array(f["temperature"]) if 'temperature' in f else None,
                # --br / --lbte
                'kappa': load(f, 'kappa'),
                'kappa_RTA': load(f, 'kappa_RTA'),
                # --wigner --br or --wigner --lbte
                'kappa_C': load(f, 'kappa_C'),
                # --wigner --br
                'kappa_P_RTA': load(f, 'kappa_P_RTA'),
                'kappa_TOT_RTA': load(f, 'kappa_TOT_RTA'),
                # --wigner --lbte
                'kappa_P_exact': load(f, 'kappa_P_exact'),
                'kappa_TOT_exact': load(f, 'kappa_TOT_exact')}
        
    return data

def validate(data,
             filepath):
    
    k   = data['kappa']
    kC  = data['kappa_C']
    kPR = data['kappa_P_RTA']
    kTR = data['kappa_TOT_RTA']
    if k is None and (kC is None or kPR is None or kTR is None):
        print(f"ERROR! Essential variables are not found in {filepath}.")
        exit(0)

def choose_temperature(temperature,
                       filepath,
                       last_temperature,
                       current_temp):
    
    if last_temperature is None or not np.array_equal(temperature, last_temperature):
        if len(temperature) == 1:
            return temperature[0]
        else:
            print(f"\n[{filepath}] List of temperatures:")
            print("   ".join(map(str, temperature)))
            return float(input("Choose temperature: "))
    
    return current_temp

def get_temp_index(temperature,
                   target_temp,
                   filepath):

    index = np.where(temperature == target_temp)[0]
    if len(index) == 0:
        print(f"ERROR! Temperature {target_temp} not found in {filepath}.")
        exit(0)
    return index[0]

def extract_row(mesh, kappa_arr, temp_index):
    
    return (mesh[0], mesh[1], mesh[2],
            kappa_arr[temp_index, 0], kappa_arr[temp_index, 1], kappa_arr[temp_index, 2], kappa_arr[temp_index, 3], kappa_arr[temp_index, 4], kappa_arr[temp_index, 5])

def write_dat(filename, rows, display=False):
    
    header1 = "# Thermal conductivity(W/m-K) vs Q-Mesh"
    header2 = "#  Q_x   Q_y   Q_z        xx          yy          zz          yz          xz          xy"
    row_fmt  = (" {0:>5.0f} {1:>5.0f} {2:>5.0f}"
                "  {3:>10.3f}  {4:>10.3f}  {5:>10.3f}  {6:>10.3f}  {7:>10.3f}  {8:>10.3f}")

    with open(filename, 'w') as o:
        o.write(header1 + "\n")
        o.write(header2 + "\n")
        for item in rows:
            o.write(row_fmt.format(*item) + "\n")
        o.write("\n")

    if display:
        print(filename)
        print(header1)
        print(header2)
        for item in rows:
            print(row_fmt.format(*item))
        print("")

def main():
    if '-h' in argv or len(argv) != 1:
        usage()

    files = find_kappa_files()
    collected = {'kappa': [],
                 'kappa_RTA': [],
                 'kappa_C': [],
                 'kappa_P_RTA': [],
                 'kappa_TOT_RTA' : [],
                 'kappa_P_exact'   : [],
                 'kappa_TOT_exact' : []}

    last_temperature = None
    target_temp      = None

    for filepath in files:
        data = read_HDF5(filepath)
        validate(data, filepath)
        temperature = data['temperature']
        target_temp = choose_temperature(temperature, filepath, last_temperature, target_temp)
        last_temperature = temperature
 
        temp_index = get_temp_index(temperature, target_temp, filepath)
        mesh = data['mesh']
 
        for key in collected:
            if data[key] is not None:
                collected[key].append(extract_row(mesh, data[key], temp_index))

    output_map = {'kappa': ('KappaVsMesh.dat', True),
                  'kappa_RTA': ('Kappa_RTAVsMesh.dat', False),
                  'kappa_C': ('Kappa_CVsMesh.dat', False),
                  'kappa_P_RTA': ('Kappa_P_RTAVsMesh.dat', True),
                  'kappa_TOT_RTA': ('Kappa_TOT_RTAVsMesh.dat', False),
                  'kappa_P_exact': ('Kappa_P_exactVsMesh.dat', False),
                  'kappa_TOT_exact': ('Kappa_TOT_exactVsMesh.dat', False)}
 
    for key, (filename, display) in output_map.items():
        rows = collected[key]
        if rows:
            write_dat(filename, np.array(rows), display=display)

if __name__ == "__main__":
    main()
