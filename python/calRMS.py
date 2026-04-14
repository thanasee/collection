#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():

    text = """
Usage: calRMS.py <input POSCAR> <input FORCE_CONSTANTS>

This script calculate RMS of 2nd order of IFCs
and compare with distance between atoms from POSCAR/CONTCAR files

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

def mapping_elements(elements,
                     atom_counts,
                     positions_cartesian,
                     positions_direct,
                     species,
                     selective_dynamics,
                     flags):
    
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
     
    # Preserve insertion order for unique elements
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
                break
            print("ERROR! The species do not match the unique elements. Try again.")
     
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
        new_species = np.array(sort_species)
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

def read_FORCE_CONSTANTS(filepath,
                         total_atoms):

    with open(filepath, 'r') as f:
        force_lines = f.readlines()

    total_symmetry, force_total_atoms = map(int, force_lines[0].split())

    if force_total_atoms != total_atoms:
        print("ERROR! Total atoms not match.")
        exit(1)

    pair_list = []
    rms = []
    line_index = 1

    for _ in range(total_symmetry):
        for _ in range(force_total_atoms):
            pair_list.append(force_lines[line_index].split())
            line_index += 1

            rms.append(np.sqrt(np.mean([float(x) ** 2
                                        for i in range(3)
                                        for x in force_lines[line_index + i].split()])))
            line_index += 3

    return {"total_symmetry": total_symmetry,
            "pair_list": pair_list,
            "rms": rms}

def compute_image_offsets(lattice_matrix):
    
    klm = np.array([[k, l, m] for k in range(-1, 2)
                               for l in range(-1, 2)
                               for m in range(-1, 2)])   # (27, 3)
    
    return np.dot(klm, lattice_matrix)

def calculate_distance_rms(lattice_matrix,
                           total_atoms,
                           positions_cartesian,
                           image_offsets,
                           pair_list,
                           rms,
                           labels):

    distance_rms = []
    for index, s in enumerate(range(0, len(pair_list), total_atoms)):
        select = int(pair_list[s][0]) - 1
        reference_position_cartesian = positions_cartesian[select]

        # Mask out the selected atom
        mask = np.arange(total_atoms) != select
        other_positions_cartesian = positions_cartesian[mask]                                          # (N-1, 3)
        other_labels = [labels[i] for i in range(total_atoms) if i != select]
        other_rms = [rms[i + index * total_atoms] for i in range(total_atoms) if i != select]

        # Displacement vectors: (N-1, 3)
        diff = other_positions_cartesian - reference_position_cartesian

        # Add all image offsets: (N-1, 1, 3) + (1, 27, 3) → (N-1, 27, 3)
        diff_images = diff[:, np.newaxis, :] + image_offsets[np.newaxis, :, :]

        # Minimum image distances: (N-1,)
        min_distances = np.linalg.norm(diff_images, axis=2).min(axis=1)

        for label, distance, r in zip(other_labels, min_distances, other_rms):
            distance_rms.append((labels[select], label, distance, r))

    # Sort by distance
    distance_rms.sort(key=lambda x: x[2])

    return distance_rms

def write_output(elements,
                 distance_rms):

    element_pairs = [(elements[i], elements[j])
                     for i in range(len(elements))
                     for j in range(i, len(elements))]

    for pair in element_pairs:
        filename = f"RMS_{pair[0]}-{pair[1]}.dat"
        with open(filename, 'w') as o:
            o.write("# Distance vs RMS of 2nd IFCs\n")
            o.write("#   Distance      RMS\n")
            for item in distance_rms:
                if ((pair[0] in item[0] and pair[1] in item[1]) or
                        (pair[1] in item[0] and pair[0] in item[1])):
                    o.write(f"  {item[2]:>12.8f}  {item[3]:>12.8f}\n")

def main():
    if '-h' in argv or len(argv) != 3:
        usage()
    
    poscar = read_POSCAR(argv[1])
    mapping = mapping_elements(poscar["elements"],
                               poscar["atom_counts"],
                               poscar["positions_cartesian"],
                               poscar["positions_direct"],
                               poscar["species"],
                               poscar["selective_dynamics"],
                               poscar["flags"])
    force_constants = read_FORCE_CONSTANTS(argv[2],
                                           poscar["total_atoms"])
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])
    image_offsets = compute_image_offsets(poscar["lattice_matrix"])
    distance_rms = calculate_distance_rms(poscar["lattice_matrix"],
                                          poscar["total_atoms"],
                                          mapping["positions_cartesian"],
                                          image_offsets,
                                          force_constants["pair_list"],
                                          force_constants["rms"],
                                          labels)
    write_output(mapping["elements"],
                 distance_rms)

if __name__ == "__main__":
    main()
