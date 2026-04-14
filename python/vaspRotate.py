#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: vaspRotate.py <input> <output>

This script rotate atoms in POSCAR/CONTCAR file.

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
        o.write("Generated by vaspRotate.py code\n")
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

def rotation_matrix():
    
    text = """
Choices of rotation axis
1) X Axis
2) Y Axis
3) Z Axis
4) An Axis Passing Through Specified Vector"""
    print(text)
    
    while True:
        option_axis = input("Enter axis: ")
        if option_axis.isdecimal() and option_axis != '0':
            if option_axis in ['1', '2', '3']:
                axis = int(option_axis) - 1
                u = np.array([1. if i == axis else 0. for i in range(3)])
                break
            elif option_axis == '4':
                print("ex. 1 0 0 means the rotation axis is x axis")
                while True:
                    v = input("Enter the vector: ")
                    if all(vi.lstrip('-').replace('.', '').isdigit() for vi in v.split()):
                        break
                    else:
                        print("ERROR! Wrong input vector")
                u = np.array([float(vi) for vi in v.split()])
                u /= np.linalg.norm(u)
                break
            else:
                print("ERROR!! Choose again")

    # Choose the rotation degree
    while True:
        input_degree = input("Input rotation degree: ")
        if input_degree.lstrip('-').replace('.', '').isdigit():
            break
        else:
            print("ERROR! Wrong input degree")
    degree = np.radians(float(input_degree))

    # define trigonometry functions
    sin = np.sin(degree)
    cos = np.cos(degree)

    # Matrix of rotation
    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)
    
    return rotate

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

def select_pivot(lattice_matrix,
                 total_atoms,
                 positions_cartesian,
                 species):
    
    print("""
Choices of type of material:
1) molecules
2) 2D/3D materials""")

    while True:
        input_type = input("Enter choice: ")
        if input_type in ['1', '2']:
            break
        print("ERROR! Wrong input")

    if input_type == '1':
        print("""
Method for selecting the pivot point of molecule
1) center of molecule
2) position of atom in molecule
3) Custom""")

        while True:
            option = input("Enter method: ")
            if option == '1':
                ref_point = np.mean(positions_cartesian, axis=0)
                break
            elif option == '2':
                for j in range(total_atoms):
                    print(f"{species[j]} atom : {j + 1:>3.0f}")
                while True:
                    select_atom = input(f"Select the atom as the pivot point ({1:>3} to {total_atoms:>3}): ")
                    if select_atom.isdecimal() and 0 < int(select_atom) <= total_atoms:
                        break
                    else:
                        print("Wrong No. of atom")
                ref_point = positions_cartesian[int(select_atom) - 1]
                break
            elif option == '3':
                point = []
                for i in ('a', 'b', 'c'):
                    while True:
                        p = input(f"Enter position in {i} direction (direct): ")
                        if p.lstrip('-').replace('.', '').isdigit():
                            break
                    point.append(float(p))
                ref_point = np.dot(np.array(point), lattice_matrix)
                break
            else:
                print("ERROR!! Choose method again")
        return input_type, None, ref_point

    else:
        selected_atoms = select_index(total_atoms, species)
        ref_point = np.mean(positions_cartesian[selected_atoms], axis=0)
        return input_type, selected_atoms, ref_point

def rotate_atoms(lattice_matrix,
                 total_atoms,
                 positions_cartesian,
                 species,
                 rotate_matrix):
    
    input_type, selected_atoms, ref_point = select_pivot(lattice_matrix, total_atoms, positions_cartesian, species)

    if input_type == '1':
        new_positions_cartesian = np.dot(positions_cartesian - ref_point, rotate_matrix.T) + ref_point
    else:
        new_positions_cartesian = np.copy(positions_cartesian)
        new_positions_cartesian[selected_atoms] = (np.dot(positions_cartesian[selected_atoms] - ref_point, rotate_matrix.T) + ref_point)

    return new_positions_cartesian

def main():
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
    
    unrotate = read_POSCAR(argv[1])
    rotate = rotation_matrix()
    rotate_positions_cartesian = rotate_atoms(unrotate["lattice_matrix"],
                                              unrotate["total_atoms"],
                                              unrotate["positions_cartesian"],
                                              unrotate["species"],
                                              rotate)
    rotate_positions_direct = cartesian_to_direct(unrotate["lattice_matrix"],
                                                  rotate_positions_cartesian)
    mapping = mapping_elements(unrotate["elements"],
                               unrotate["atom_counts"],
                               rotate_positions_cartesian,
                               rotate_positions_direct,
                               unrotate["species"],
                               unrotate["selective_dynamics"],
                               unrotate["flags"])
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])
    write_POSCAR(argv[2],
                 unrotate["lattice_matrix"],
                 mapping["elements"],
                 mapping["atom_counts"],
                 mapping["positions_direct"],
                 unrotate["selective_dynamics"],
                 mapping["flags"],
                 labels)
    
    print("")

if __name__ == "__main__":
    main()
