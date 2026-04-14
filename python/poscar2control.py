#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: poscar2control.py <input>

This script convert POSCAR format to CONTROL format for ShengBTE.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

def read_POSCAR(filepath):
    
    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)
    
    with open(filepath, 'r') as poscar:
        lines = poscar.readlines()
    
    if len(lines[1].split()) == 1:
        raw_scale = float(lines[1])
        raw_lattice_matrix = np.array([[float(x) for x in line.split()]
                                       for line in lines[2:5]])
        if raw_scale < 0:
            volume = np.abs(np.linalg.det(raw_lattice_matrix))
            scale = np.cbrt(np.abs(raw_scale) / volume)
        elif raw_scale == 0:
            print("ERROR! The scaling factor must be not zero.")
            exit(1)
        else:
            scale = raw_scale
        lattice_matrix = raw_lattice_matrix * scale
    elif len(lines[1].split()) == 3:
        scale = np.array(list(map(float, lines[1].split())))
        lattice_matrix = np.array([[float(x) * scale[i] for i, x in enumerate(line.split())]
                                   for line in lines[2:5]])
    else:
        print("ERROR! The scaling factor must be 1 or 3 components.")
        exit(1)
    
    elements = []
    is_number = lines[5].split()[0].isdecimal()
    if is_number:
        # VASP4 format: no element line -> prompt user
        for i in range(len(lines[5].split())):
            while True:
                name = input(f"Enter the name of species No. {i + 1:>3}: ").strip()
                if name.isalpha():
                    break
                else:
                    print("The name of species must be alphabetic characters only.")
            elements.append(name)
        atom_counts = [int(x) for x in lines[5].split()]
        selective_dynamics = lines[6].lower().startswith('s')
        position_start = 8 if selective_dynamics else 7
    else:
        # VASP5 format: element symbols present
        raw_elements = lines[5].split()
        for name in raw_elements:
            elements.append(name.split('/')[0].split('_')[0])
        atom_counts = [int(x) for x in lines[6].split()]
        selective_dynamics = lines[7].lower().startswith('s')
        position_start = 9 if selective_dynamics else 8

    # Read atomic positions
    total_atoms = sum(atom_counts)
    position_stop = position_start + total_atoms

    positions = np.array([[float(x) for x in lines[i].split()[:3]]
                          for i in range(position_start, position_stop)])

    species = [x for i, x in enumerate(elements)
               for _ in range(atom_counts[i])]

    flags = None
    if selective_dynamics:
        flags = np.array([[x for x in lines[i].split()[3:6]]
                          for i in range(position_start, position_stop)])

    # Detect coordinate type
    is_direct = lines[position_start - 1].strip().lower().startswith('d')
    if is_direct:
        # If direct coordinate then compute cartesian coordinate
        positions_direct = positions % 1.0
        positions_cartesian = direct_to_cartesian(lattice_matrix, positions_direct)
    else:
        # If cartesian coordinate then direct coordinate
        positions_cartesian = positions * scale
        positions_direct = cartesian_to_direct(lattice_matrix, positions_cartesian)
    
    return {"lattice_matrix": lattice_matrix,
            "elements": elements,
            "atom_counts": atom_counts,
            "total_atoms": total_atoms,
            "positions_cartesian": positions_cartesian,
            "positions_direct": positions_direct,
            "species": species,
            "selective_dynamics": selective_dynamics,
            "flags": flags if selective_dynamics else None}

def direct_to_cartesian(lattice_matrix,
                        positions_direct):
    
    positions = positions_direct % 1.0
    positions_cartesian = np.dot(positions, lattice_matrix)
    
    return positions_cartesian

def cartesian_to_direct(lattice_matrix,
                        positions_cartesian):
    
    positions_direct = np.dot(positions_cartesian, np.linalg.inv(lattice_matrix)) % 1.0
    
    return positions_direct

def get_vacuum_index():
    
    while True:
        dim = input("2D materials or not (Y/N): ").strip().lower()
        if dim[0] == 'y':
            while True:
                vacuum = input("Enter vacuum direction (a/b/c): ").strip().lower()
                if vacuum in ("a", "b", "c"):
                    break
                else:
                    print("Input must be a, b, or c!")
            return {"a": 0, "b": 1, "c": 2}[vacuum]
        elif dim[0] == 'n':
            return None
        else:
            print("Input must be Y or N!")

def get_ngrid(reciprocal_matrix,
              index):
 
    while True:
        spacing = input("Enter q-point spacing in 2π/Å: ").strip()
        try:
            qspacing = float(spacing) * 2 * np.pi
            if qspacing > 0:
                ngrid = []
                for i in range(3):
                    if i == index:
                        ngrid.append(1)
                    else:
                        n = max(1, int(np.round(np.linalg.norm(reciprocal_matrix[i]) / qspacing)))
                        ngrid.append(n)
                return ngrid
            else:
                print("Spacing must be positive.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_supercell_matrix():
 
    while True:
        supercell_matrix = input("Enter Supercell Matrix (3 elements): ")
        try:
            check_supercell = np.array(list(map(int, supercell_matrix.split())))
            if len(check_supercell) == 3 and all(val > 0 for val in check_supercell):
                return supercell_matrix
            else:
                print("Input must be 3 components and positive integers!")
        except ValueError:
            print("Invalid input. Please enter integer numbers separated by spaces.")

def get_phonon_flags():
 
    while True:
        n_phonon = input("Enter n phonon process: ")
        try:
            n_phonon = int(n_phonon)
            if n_phonon == 3:
                return """        convergence=.true.
        nthreads=-1
&end"""
            elif n_phonon == 4:
                while True:
                    device = input("Enter parallel devices (CPU/GPU): ").strip().upper()
                    if device[0] == 'C':
                        print("Your parallel devices use CPU.")
                        convergence = '.true.'
                        break
                    elif device[0] == 'G':
                        print("Your parallel devices use GPU.")
                        convergence = '.false.'
                        break
                    else:
                        print("Input must be CPU or GPU!")
                return f"""        convergence={convergence}
        four_phonon=.true.
        four_phonon_iteration={convergence}
&end"""
            else:
                print("Input must be 3 or 4!")
        except ValueError:
            print("Invalid input. Please enter integer numbers.")

def write_CONTROL(filepath,
                  lattice_matrix,
                  elements,
                  atom_counts,
                  positions_direct,
                  ngrid,
                  supercell_matrix,
                  phonon_flags):
 
    total_atoms = sum(atom_counts)
    elements_str = " ".join([f'"{element}"' for element in elements])
    types = " ".join([str(i + 1) for i, count in enumerate(atom_counts)
                      for _ in range(count)])
 
    control = "&allocations"
    control += f"""\n        nelements= {len(elements)},
        natoms= {total_atoms},
        ngrid(:)= {ngrid[0]} {ngrid[1]} {ngrid[2]}
&end
&crystal
        lfactor={.1:.16f},
"""
 
    for i, vec in enumerate(lattice_matrix, 1):
        control += f"        lattvec(:,{i})= {vec[0]:.16f}  {vec[1]:.16f}  {vec[2]:.16f},\n"
 
    control += f"""        elements= {elements_str}
        types= {types},
"""
 
    for i, position in enumerate(positions_direct, 1):
        comma = "," if i < total_atoms else ""
        control += f"        positions(:,{i})= {position[0]:.16f}  {position[1]:.16f}  {position[2]:.16f}{comma}\n"
 
    control += f"""        scell(:)= {supercell_matrix}
&end
&parameters
        T=300.
        scalebroad=1.0
&end
&flags
        nonanalytic=.false.
        isotopes=.true.
        autoisotopes=.true.
        nanowire=.false.
        onlyharmonic=.false.
"""
    control += phonon_flags
 
    with open(filepath, 'w') as o:
        o.write(control)

def main():
 
    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()
 
    poscar = read_POSCAR(argv[1])
 
    reciprocal_matrix = 2. * np.pi * np.linalg.inv(poscar["lattice_matrix"]).T
    vac_idx = get_vacuum_index()
    ngrid = get_ngrid(reciprocal_matrix, vac_idx)
    supercell_matrix = get_supercell_matrix()
    phonon_flags = get_phonon_flags()
 
    output_file = "CONTROL.initial"
    write_CONTROL(output_file,
                  poscar["lattice_matrix"],
                  poscar["elements"],
                  poscar["atom_counts"],
                  poscar["positions_direct"],
                  ngrid,
                  supercell_matrix,
                  phonon_flags)
 
    print(f"Written: {output_file}")

if __name__ == "__main__":
    main()
