#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: vaspSupercell.py <input> <output>
 
This script supports VASP5 structure file format (i.e. POSCAR)
for extending a structure file from a unit cell to a supercell.
 
The expansion matrix can be specified as:
  3 components  ->  diagonal matrix  S_xx S_yy S_zz
  9 components  ->  full 3×3 matrix  S_xx S_xy S_xz  S_yx S_yy S_yz  S_zx S_zy S_zz
 
This script was inspired by Jiraroj T-Thienprasert
and developed by Thanasee Thanasarnsurapong.
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

def check_elements(elements):
    
    unique_elements = list(dict.fromkeys(elements))
     
    if len(elements) != len(unique_elements):
        print("\nFound duplicated elements in POSCAR!")
        print("Unique elements: [" + " ".join(unique_elements) + "]")
        while True:
            sort_elements = input("Enter the desired element order (separate by space): ").split()
            if len(sort_elements) == 0:
                print("Warning! Empty input — using default unique element order.")
                sort_elements = unique_elements.copy()
            if (len(sort_elements) == len(unique_elements) and
                    set(sort_elements) == set(unique_elements)):
                return sort_elements
            print("ERROR! The species do not match the unique elements. Try again.")
    else:
        return None

def mapping_elements(elements,
                     atom_counts,
                     positions_cartesian,
                     positions_direct,
                     species,
                     selective_dynamics,
                     flags,
                     sort_elements=None):
    
    new_elements = elements.copy()
    new_atom_counts = atom_counts.copy()
    new_positions_cartesian = positions_cartesian.copy()
    new_positions_direct = positions_direct.copy()
    new_species = species.copy()
    new_flags = flags.copy() if selective_dynamics else None
    
    elements_positions_cartesian = {}
    elements_positions_direct = {}
    elements_species = {}
    elements_flags = {} if selective_dynamics else None
    position_index = 0
    for element, count in zip(elements, atom_counts):
        elements_positions_cartesian.setdefault(element, []).extend(
            new_positions_cartesian[position_index:position_index + count])
        elements_positions_direct.setdefault(element, []).extend(
            new_positions_direct[position_index:position_index + count])
        elements_species.setdefault(element, []).extend(
            new_species[position_index:position_index + count])
        if selective_dynamics:
            elements_flags.setdefault(element, []).extend(
                new_flags[position_index:position_index + count])
        position_index += count
     
    if sort_elements is None:
        sort_elements = check_elements(elements)
    
    if sort_elements is not None:     
        sort_positions_cartesian = []
        sort_positions_direct = []
        sort_species = []
        sort_flags = [] if selective_dynamics else None
        sort_atom_counts = []
        for element in sort_elements:
            sort_positions_cartesian.extend(elements_positions_cartesian[element])
            sort_positions_direct.extend(elements_positions_direct[element])
            sort_species.extend(elements_species[element])
            if selective_dynamics:
                sort_flags.extend(elements_flags[element])
            sort_atom_counts.append(len(elements_positions_direct[element]))
     
        new_positions_cartesian = np.array(sort_positions_cartesian, dtype=float)
        new_positions_direct = np.array(sort_positions_direct, dtype=float)
        new_species = list(sort_species)
        if selective_dynamics:
            new_flags = np.array(sort_flags)
        new_atom_counts = sort_atom_counts
        new_elements = sort_elements

    return {"elements": new_elements,
            "atom_counts": new_atom_counts,
            "positions_cartesian": new_positions_cartesian,
            "positions_direct": new_positions_direct,
            "species": new_species,
            "flags": new_flags if selective_dynamics else None}

def define_labels(elements,
                  atom_counts):
    
    digits = len(str(max(atom_counts))) + 1
    labels = [f"{symbol}{str(counter).zfill(digits)}" for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]
    
    return labels

def write_POSCAR(filepath,
                 lattice_matrix,
                 elements,
                 atom_counts,
                 positions_direct,
                 selective_dynamics,
                 flags,
                 labels):
    
    with open(filepath, 'w') as o:
        o.write("Generated by vaspSupercell.py code\n")
        o.write(f"   {1.0:.14f}\n")
        for lattice in lattice_matrix:
            o.write(f"   {lattice[0]:20.16f}  {lattice[1]:20.16f}  {lattice[2]:20.16f}\n")
        o.write("   " + "    ".join(elements) + " \n")
        o.write("     " + "    ".join(map(str, atom_counts)) + "\n")
        if selective_dynamics:
            o.write("Selective dynamics\n")
        o.write("Direct\n")
        if selective_dynamics:
            for position, flag, label in zip(positions_direct, flags, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {flag[0]:s}   {flag[1]:s}   {flag[2]:s}" + f"   {label:>6s}\n")
        else:
            for position, label in zip(positions_direct, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {label:>6s}\n")

def input_expansion():
    
    text = """
Expansion matrix components:
  3 components  ->  S_xx S_yy S_zz  (diagonal)
  9 components  ->  S_xx S_xy S_xz  S_yx S_yy S_yz  S_zx S_zy S_zz  (full 3×3)
Enter expansion matrix (separate by space):"""
    print(text)

    while True:
        expansion = input()
        try:
            values = np.array(list(map(int, expansion.split())))
            if len(values) == 3:
                expansion_matrix = np.diag(values)
            elif len(values) == 9:
                expansion_matrix = values.reshape(3, 3)
            else:
                print("Input must be 3 or 9 compenents!")
                continue
        except ValueError:
            print("Invalid input. Please enter integer numbers separated by spaces.")
            continue
        det = np.linalg.det(expansion_matrix)
        det_int = int(round(det))
        if det > 1e-10 and abs(det - det_int) < 1e-6 and det_int > 0:
            break
        elif det <= 1e-10:
            print("Invalid expansion matrix: determinant must be a positive integer.")
        else:
            print("Invalid expansion matrix: determinant is not an integer. "
"Please enter integer components that yield a positive integer determinant.")
    
    return expansion_matrix, det_int

def build_supercell(expansion_matrix,
                    replicas,
                    lattice_matrix,
                    atom_counts,
                    total_atoms,
                    positions_cartesian,
                    species,
                    selective_dynamics,
                    flags):
    
    # Expansion of lattice matrix
    new_lattice_matrix = np.dot(expansion_matrix, lattice_matrix)
    
    # Generate lattice grid points inside the supercell
    # Use the 8 corners of the unit supercell box to bound the search range
    corners = np.array([[i, j, k]
                        for i in range(2) for j in range(2) for k in range(2)])
    corner_transformed = np.dot(corners, expansion_matrix)
    min_points = np.min(corner_transformed, axis=0).astype(int)
    max_points = np.max(corner_transformed, axis=0).astype(int) + 1

    # Generate all combinations of i, j, k within the given expansion matrix
    all_points = np.array([[i, j, k] for i in range(min_points[0], max_points[0])
                                for j in range(min_points[1], max_points[1])
                                for k in range(min_points[2], max_points[2])])

    # Keep only points whose fractional coordinates in the supercell are in [0, 1)
    adj = np.round(np.linalg.inv(expansion_matrix) * replicas).astype(int)
    frac_points = np.dot(all_points, adj)
    mask = (np.all(frac_points >= 0, axis=1) &
            np.all(frac_points <  replicas, axis=1))
    grid_points = all_points[mask]
     
    if len(grid_points) != replicas:
        print(f"WARNING: Expected {replicas} grid points but found {len(grid_points)}. "
"Check your expansion matrix.")
    
    # Generate new atomic positions
    # positions are in Å; grid_points are in primitive-cell lattice coordinates
    # Cartesian displacement for each grid point: dot(grid_point, lattice_matrix)
    grid_cartesian = np.dot(grid_points, lattice_matrix)  # shape: (n_replicas, 3)
     
    # Broadcast: (n_atoms, 1, 3) + (1, n_replicas, 3) → (n_atoms, n_replicas, 3)
    new_positions_cartesian = (positions_cartesian[:, np.newaxis, :] + grid_cartesian[np.newaxis, :, :]).reshape(-1, 3)
    new_species = np.repeat(species, len(grid_points)).tolist()
    
    if selective_dynamics:
        new_flags = np.tile(flags[:, np.newaxis, :], (1, len(grid_points), 1)).reshape(-1, 3)

    # New atom counts per element
    new_atom_counts = [count * replicas for count in atom_counts]
    new_total_atoms = total_atoms * replicas
    
    new_positions_direct = cartesian_to_direct(new_lattice_matrix, new_positions_cartesian)
    
    return {"lattice_matrix": new_lattice_matrix,
            "atom_counts": new_atom_counts,
            "total_atoms": new_total_atoms,
            "positions_direct": new_positions_direct,
            "positions_cartesian": new_positions_cartesian,
            "species": new_species,
            "flags": new_flags if selective_dynamics else None}

def main():
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
    
    unitcell = read_POSCAR(argv[1])
    expansion_matrix, replicas = input_expansion()
    supercell = build_supercell(expansion_matrix,
                                replicas,
                                unitcell["lattice_matrix"],
                                unitcell["atom_counts"],
                                unitcell["total_atoms"],
                                unitcell["positions_cartesian"],
                                unitcell["species"],
                                unitcell["selective_dynamics"],
                                unitcell["flags"])
    mapping = mapping_elements(unitcell["elements"],
                               supercell["atom_counts"],
                               supercell["positions_cartesian"],
                               supercell["positions_direct"],
                               supercell["species"],
                               unitcell["selective_dynamics"],
                               supercell["flags"])
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])
    write_POSCAR(argv[2],
                 supercell["lattice_matrix"],
                 mapping["elements"],
                 mapping["atom_counts"],
                 mapping["positions_direct"],
                 unitcell["selective_dynamics"],
                 mapping["flags"],
                 labels)
    
    # Summary
    print(f"\nSupercell written to: {argv[2]}")
    print(f"Expansion matrix determinant: {replicas}")
    print("-" * 39)
    print("  Element  |  Unit cell  |  Supercell")
    print("-" * 39)
    for element, orig, new in zip(mapping["elements"], unitcell["atom_counts"], mapping["atom_counts"]):
        print(f"  {element:<9}|  {orig:<11}|  {new}")
    print("-" * 39)
    print(f"  Total    |  {unitcell['total_atoms']:<11}|  {supercell['total_atoms']}")
    print("-" * 39 + "\n")

if __name__ == "__main__":
    main()
