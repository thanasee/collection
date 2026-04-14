#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: vaspShift.py <input> <output>

This script support VASP5 Structure file format (i.e. POSCAR) 
for shifting a structure file:
  molecule (0D)  ->  shift to center
  nanowire (1D)  ->  shift to origin in extend direction and center in other direction
  sheet (2D)     ->  shift to center in vacuum direction and center in other direction
  bulk (3D)      ->  shift to origin
  adsorbate      ->  shift to origin in XY

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
        o.write("Generated by vaspShift.py code\n")
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

def unwrap(positions_direct):
    
    reference = np.copy(positions_direct[0])
    delta = positions_direct - reference
    delta -= np.round(delta)
    
    return reference, reference + delta
    
def get_direction(prompt):
    
    print(f"""
Input the direction index of {prompt} direction (1 to 3):
1) x direction
2) y direction
3) z direction""")
    while True:
        try:
            idx = int(input()) - 1
            if 0 <= idx < 3:
                return idx
            print("ERROR! Directions must be between 1 and 3. Try again.")
        except ValueError:
            print("ERROR! Must enter a number. Try again.")

def get_adsorbent_atoms(total_atoms, species):
    
    print(f"""
Input element-symbol and/or atom-indexes of adsorbent ({1:>3} to {total_atoms:>3})
(Free-format input, e.g., 1 3 1-4 C H all)""")
    while True:
        adsorbent_atoms = []
        for adsorbent in input().split():
            if 'all' in adsorbent:
                adsorbent_atoms.extend(range(total_atoms))
                break
            if adsorbent.isnumeric() or '-' in adsorbent:
                if '-' in adsorbent:
                    start, end = map(int, adsorbent.split('-'))
                    adsorbent_atoms.extend(range(start - 1, end))
                else:
                    adsorbent_atoms.append(int(adsorbent) - 1)
            else:
                adsorbent_atoms.extend([i for i, label in enumerate(species) if label == adsorbent])
        if len(adsorbent_atoms) > total_atoms or not all(0 <= idx < total_atoms for idx in adsorbent_atoms):
            print("Wrong input atom-indexes !TRY AGAIN!")
        else:
            return adsorbent_atoms

def shift_molecule(positions_direct):
    
    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    
    return (unwrapped - center + 0.5) % 1.0

def shift_wire(positions_direct):
    
    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    extend = get_direction("extend")
    periodic = [i for i in range(3) if i != extend]
    new = np.copy(unwrapped)
    new[:, extend]   = unwrapped[:, extend] - reference[extend]
    new[:, periodic] = unwrapped[:, periodic] - center[periodic] + 0.5
    
    return new % 1.0

def shift_sheet(positions_direct):
    
    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    vacuum = get_direction("vacuum")
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5
    
    return new % 1.0

def shift_bulk(positions_direct):
    reference, unwrapped = unwrap(positions_direct)
    return (unwrapped - reference) % 1.0

def shift_special(total_atoms,
                  positions_direct,
                  species):
    
    adsorbent_atoms = get_adsorbent_atoms(total_atoms, species)
    reference, unwrapped = unwrap(positions_direct)
    # re-unwrap around adsorbate reference
    ref_ads = np.copy(positions_direct[adsorbent_atoms[0]])
    delta = positions_direct - ref_ads
    delta -= np.round(delta)
    unwrapped = ref_ads + delta
    center = np.mean(unwrapped[adsorbent_atoms], axis=0)
    new = np.copy(unwrapped)
    new[:, :2] = unwrapped[:, :2] - center[:2] + 0.5
    new[:, 2]  = unwrapped[:, 2]
    
    return new % 1.0

def shift(total_atoms,
          positions_direct,
          species):
    
    print("""
Choices of type of material
  0) 0D (molecule)   -> shift all atoms to center
  1) 1D (wire)       -> origin in extend direction, center in other
  2) 2D (sheet)      -> origin in periodic, center in vacuum
  3) 3D (bulk)       -> shift all atoms to origin
  4) Special!        -> adsorbate XY center, Z free""")

    dispatch = {
        '0': lambda: shift_molecule(positions_direct),
        '1': lambda: shift_wire(positions_direct),
        '2': lambda: shift_sheet(positions_direct),
        '3': lambda: shift_bulk(positions_direct),
        '4': lambda: shift_special(total_atoms, positions_direct, species),
    }

    while True:
        mode = input("Enter choice: ")
        if mode in dispatch:
            return dispatch[mode]()
        elif mode.isdigit():
            print("ERROR!! Must choose type of material")
        else:
            print("ERROR!! Choose again")

def main():
    if '-h' in argv or '--help' in argv or len(argv) != 3:
        usage()
    
    unshift = read_POSCAR(argv[1])
    shift_positions_direct = shift(unshift["total_atoms"],
                                   unshift["positions_direct"],
                                   unshift["species"])
    shift_positions_cartesian = direct_to_cartesian(unshift["lattice_matrix"],
                                                    shift_positions_direct)
    mapping = mapping_elements(unshift["elements"],
                               unshift["atom_counts"],
                               shift_positions_cartesian,
                               shift_positions_direct,
                               unshift["species"],
                               unshift["selective_dynamics"],
                               unshift["flags"])
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])
    write_POSCAR(argv[2],
                 unshift["lattice_matrix"],
                 mapping["elements"],
                 mapping["atom_counts"],
                 mapping["positions_direct"],
                 unshift["selective_dynamics"],
                 mapping["flags"],
                 labels)
    
    print("")

if __name__ == "__main__":
    main()
