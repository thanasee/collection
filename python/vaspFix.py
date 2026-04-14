#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: vaspFix.py <input> <output>

This script support VASP5 Structure file format (i.e. POSCAR) 
for fix atom in Structure file.

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
        o.write("Generated by vaspFix.py code\n")
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

def refix(total_atoms,
          selective_dynamics,
          flags):
    
    # Ensure selective dynamics flags are initialized
    skip = False
    if not selective_dynamics:
        flags = np.full((total_atoms, 3), 'T')
    else:
        while True:
            print("""
Selective Dynamics already present in input
  Y) Reset all flags to T T T, then re-apply fixes
  A) Keep existing flags and add more fixes on top
  N) Keep existing flags as-is and skip to output""")
            option_refix = input("Choice (Y/A/N): ").strip().upper()
            if option_refix == 'Y':
                flags = np.full((total_atoms, 3), 'T')
                break
            elif option_refix == 'A':
                break  # keep flags, proceed to fix mode selection
            elif option_refix == 'N':
                skip = True  # skip selection, jump straight to output
                break
            else:
                print("ERROR! Enter Y, A, or N.")
    return flags, skip

def select_index(total_atoms,
                 species):
    
    print(f"""
Input element-symbol and/or atom-indexes to choose ({1:>3} to {total_atoms:>3})
(Free-format input, e.g., 1 3 1-4 C H all)""")
    while True:
        selected_atoms = []
        input_select = input().split()
 
        for select in input_select:
            if 'all' in select:
                selected_atoms.extend(range(total_atoms))
                break
            if select.isnumeric() or '-' in select:
                if '-' in select:
                    start, end = map(int, select.split('-'))
                    selected_atoms.extend(range(start - 1, end))
                else:
                    selected_atoms.append(int(select) - 1)
            else:
                selected_atoms.extend([i for i, label in enumerate(species) if label == select])
 
        if len(selected_atoms) > total_atoms or not all(0 <= idx < total_atoms for idx in selected_atoms):
            print("Wrong input atom-indexes! TRY AGAIN!")
        else:
            break
 
    return selected_atoms

def select_radius(lattice_matrix,
                  total_atoms,
                  positions_cartesian):
    
    print("""
Choose reference point
Input atom-indexes of molecule (Free-format input, e.g., 1 3 1-4): """)
    while True:
        indices = []
        input_atoms = input().split()
 
        for atom in input_atoms:
            if '-' in atom:
                try:
                    start, end = map(int, atom.split('-'))
                    indices.extend(range(start - 1, end))
                except ValueError:
                    print("Invalid range format! Try again.")
            elif atom.isnumeric():
                indices.append(int(atom) - 1)
            else:
                print("Invalid atom-index format! Try again.")
 
        if len(indices) > total_atoms or not all(0 <= idx < total_atoms for idx in indices):
            print(f"Invalid input! Ensure indices are between 1 and {total_atoms}. Try again.")
        else:
            break
 
    reference_point = np.mean(positions_cartesian[indices], axis=0)
 
    while True:
        try:
            input_radius = float(input("Input cutoff radius in Angstrom (select atoms outside cutoff radius): "))
            if input_radius > 0:
                break
        except ValueError:
            print("ERROR! Input cutoff radius must be a positive number.")
 
    selected_atoms = []
    for index, position in enumerate(positions_cartesian):
        min_distance = np.linalg.norm(position - reference_point)
 
        # Consider periodic boundary conditions
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    offset = j * lattice_matrix[0] + k * lattice_matrix[1] + l * lattice_matrix[2]
                    candidate_distance = np.linalg.norm(position + offset - reference_point)
                    min_distance = min(min_distance, candidate_distance)
 
        if min_distance > input_radius:
            selected_atoms.append(index)
 
    return selected_atoms

def select_file(total_atoms):
    select_file = 'SELECTED_FIX_ATOMS_LIST'
    while True:
        try:
            with open(select_file, 'r') as s:
                file_lines = s.readlines()
        except FileNotFoundError:
            select_file = input("Enter your SELECTED_FIX_ATOMS_LIST path: ")
            continue
 
        flags = np.array([[x for x in line.split()[2:]] for line in file_lines[3:]])
        if len(flags) == total_atoms:
            return flags
        else:
            print("ERROR! The numbers of atom in SELECTED_FIX_ATOMS_LIST and POSCAR files not match!")
            select_file = input("Enter your SELECTED_FIX_ATOMS_LIST path: ")

def select_direction():
    
    print("""
Input the direction index to fix (1 to 3):
1) x direction
2) y direction
3) z direction
(Free-format input, e.g., 1 3 1-3 all)""")
    while True:
        fix_coordinates = []
        input_coordinates = input().split()
 
        for coordinate in input_coordinates:
            if 'all' in coordinate.lower():
                fix_coordinates = [0, 1, 2]
                break
            elif '-' in coordinate:
                try:
                    start, end = map(int, coordinate.split('-'))
                    fix_coordinates.extend(range(start - 1, end))
                except ValueError:
                    print("Invalid range format! Try again.")
            elif coordinate.isnumeric():
                fix_coordinates.append(int(coordinate) - 1)
            else:
                print("Invalid direction format! Try again.")
 
        if fix_coordinates and all(0 <= idx < 3 for idx in fix_coordinates):
            return fix_coordinates
        else:
            print("ERROR! Directions must be between 1 and 3. Try again.")
    

def fix_mode(lattice_matrix,
             total_atoms,
             positions_cartesian,
             species,
             flags):
    
    print("""
Choices of fixing atoms method
1) Atomic Indexes/Labels
2) Cutoff Radius From Reference Point
3) Read From SELECTED_ATOMS_LIST File""")
 
    while True:
        fix_mode = input("Enter choice: ")
 
        if fix_mode == '1':
            selected_atoms = select_index(total_atoms, species)
            break
        elif fix_mode == '2':
            selected_atoms = select_radius(lattice_matrix, total_atoms, positions_cartesian)
            break
        elif fix_mode == '3':
            flags = select_file(total_atoms)
            return flags
        else:
            print("ERROR!! Must choose fixing atoms method")
 
    fix_coordinates = select_direction()
 
    for atom in selected_atoms:
        for direction in fix_coordinates:
            flags[atom][direction] = 'F'
 
    return flags

def write_selected(flags,
                   labels):
    
    select_file = "SELECTED_FIX_ATOMS_LIST"
    with open(select_file, 'w') as o:
        o.write("#                Fix Status\n")
        o.write("#Index   LABEL   x  y  z\n")
        o.write("#--------------------------\n")
        for i in range(len(flags)):
            o.write(f"{i + 1:5}   {labels[i]}    {flags[i, 0]}  {flags[i, 1]}  {flags[i, 2]}\n")

def main():
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
 
    poscar = read_POSCAR(argv[1])
    flags, skip = refix(poscar["total_atoms"],
                        poscar["selective_dynamics"],
                        poscar["flags"])
    if not skip:
        flags = fix_mode(poscar["lattice_matrix"],
                         poscar["total_atoms"],
                         poscar["positions_cartesian"],
                         poscar["species"],
                         flags)
    mapping = mapping_elements(poscar["elements"],
                               poscar["atom_counts"],
                               poscar["positions_direct"],
                               True,
                               flags)
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])
    write_POSCAR(argv[2],
                 poscar["lattice_matrix"],
                 mapping["elements"],
                 mapping["atom_counts"],
                 mapping["positions_direct"],
                 True,
                 mapping["flags"],
                 labels)
    write_selected(mapping["flags"], labels)
 
if __name__ == "__main__":
    main()
