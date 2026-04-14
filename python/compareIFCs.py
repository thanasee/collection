#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
import h5py as h5

def usage():
    
    text = """
Usage: compareIFCs.py <DFT input> <MLFF input>

This script extract interatomic force constant form HDF5 files, which calculated from Phono3py package.
Autodetect the order of IFCs (2nd or 3rd).

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

def validate_file(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

def read_ifc_from_hdf5(filepath):
    
    data = {}
    with h5.File(filepath, 'r') as f:
        if 'force_constants' in f:
            data['fc2'] = np.array(f['force_constants'])
        if 'fc3' in f:
            data['fc3'] = np.array(f['fc3'])
            
    return data

def validate_shapes(dft_data,
                    mlff_data,
                    order):
    
    if dft_data[order].shape != mlff_data[order].shape:
        print(f"ERROR! Shape mismatch for {order}: "
              f"DFT {dft_data[order].shape} vs MLFF {mlff_data[order].shape}.")
        exit(1)

def get_order(data):
    
    if 'fc2' in data:
        return 'fc2'
    if 'fc3' in data:
        return 'fc3'
    
    return None

def compute_residual(dft_arr,
                     mlff_arr):
    
    return mlff_arr - dft_arr

def print_summary(label,
                  units,
                  dft_arr,
                  residual):
    
    rmse = np.sqrt(np.mean(residual ** 2))
    max_abs = np.max(np.abs(residual))
    print(f"[{label}] RMSE: {rmse:.6f} {units} | Max |residual|: {max_abs:.6f} {units}")

def write_dat_file(output_file,
                   label,
                   units,
                   dft_arr,
                   residual):
    
    with open(output_file, 'w') as o:
        o.write(f"# {label} comparison ({units})\n")
        o.write(f"#{'DFT':>18}{'MLFF - DFT':>18}\n")
        for dft, res in zip(dft_arr.flatten(), residual.flatten()):
            o.write(f"{dft:>18.8f}{res:>18.8f}\n")

def process_ifc(order,
                dft_data,
                mlff_data):
    
    ORDER_META = {'fc2': ('2nd IFCs', 'eV/Ang**2', '2ndIFCs.dat'),
                  'fc3': ('3rd IFCs', 'eV/Ang**3', '3rdIFCs.dat')}

    dft_arr = dft_data[order]
    mlff_arr = mlff_data[order]
    residual = compute_residual(dft_arr, mlff_arr)

    label, units, output_file = ORDER_META[order]

    print_summary(label, units, dft_arr, residual)
    write_dat_file(output_file, label, units, dft_arr, residual)

def main():
    
    if '-h' in argv or len(argv) < 2 or len(argv) > 3:
        usage()
    
    dft_file  = argv[1]
    validate_file(dft_file)
    mlff_file = argv[2]
    validate_file(mlff_file)
    
    dft_data  = read_ifc_from_hdf5(dft_file)
    mlff_data = read_ifc_from_hdf5(mlff_file)
    
    dft_order  = get_order(dft_data)
    mlff_order = get_order(mlff_data)
    if dft_order is None or mlff_order is None:
        print("ERROR! No recognisable IFC data found in one or both input files.")
        exit(1)
    if dft_order != mlff_order:
        print(f"ERROR! Order mismatch: DFT has {dft_order}, MLFF has {mlff_order}.")
        exit(1)
    
    
    validate_shapes(dft_data, mlff_data, dft_order)
    process_ifc(dft_order, dft_data, mlff_data)
    
if __name__ == '__main__':
    main()
