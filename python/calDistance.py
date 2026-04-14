#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():

    text = """
Usage: calDistance.py <input>

This script calculate distance between atoms from POSCAR/CONTCAR files.

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

def mapping_elements(elements,
                     atom_counts,
                     positions_cartesian,
                     positions_direct,
                     selective_dynamics,
                     flags):
    
    new_elements = elements.copy()
    new_atom_counts = atom_counts.copy()
    new_positions_cartesian = positions_cartesian.copy()
    new_positions_direct = positions_direct.copy()
    new_flags = flags.copy() if selective_dynamics else None
    
    elements_positions_cartesian = {}
    elements_positions_direct = {}
    elements_flags = {} if selective_dynamics else None
    position_index = 0
    for element, count in zip(elements, atom_counts):
        elements_positions_cartesian.setdefault(element, []).extend(
            new_positions_cartesian[position_index:position_index + count])
        elements_positions_direct.setdefault(element, []).extend(
            new_positions_direct[position_index:position_index + count])
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
        sort_flags = [] if selective_dynamics else None
        sort_atom_counts = []
        for element in sort_elements:
            sort_positions_cartesian.extend(elements_positions_cartesian[element])
            sort_positions_direct.extend(elements_positions_direct[element])
            if selective_dynamics:
                sort_flags.extend(elements_flags[element])
            sort_atom_counts.append(len(elements_positions_direct[element]))
     
        new_positions_cartesian = np.array(sort_positions_cartesian, dtype=float)
        new_positions_direct = np.array(sort_positions_direct, dtype=float)
        if selective_dynamics:
            new_flags = np.array(sort_flags)
        new_atom_counts = sort_atom_counts
        new_elements = sort_elements

    return {"elements": new_elements,
            "atom_counts": new_atom_counts,
            "positions_cartesian": new_positions_cartesian,
            "positions_direct": new_positions_direct,
            "flags": new_flags if selective_dynamics else None}

def define_labels(elements,
                  atom_counts):
    
    digits = len(str(max(atom_counts))) + 1
    labels = [f"{symbol}{str(counter).zfill(digits)}" for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]
    
    return labels

def compute_image_offsets(lattice_matrix):
    
    klm = np.array([[k, l, m] for k in range(-1, 2)
                               for l in range(-1, 2)
                               for m in range(-1, 2)])
    
    return np.dot(klm, lattice_matrix)

def min_image_distance(position_i,
                       position_j,
                       image_offsets):
    
    diff = position_j - position_i
    diff_offset = diff[np.newaxis, :] + image_offsets
    
    return np.linalg.norm(diff_offset, axis=1).min()

def min_image_distances(position_reference,
                        positions_others,
                        image_offsets):
    
    diff = positions_others - position_reference
    diff_offset = diff[:, np.newaxis, :] + image_offsets[np.newaxis, :, :]
    
    return np.linalg.norm(diff_offset, axis=2).min(axis=1)

def parse_group(prompt,
                total_atoms,
                species,
                allow_all=True):
    
    print(prompt)
    while True:
        group = []
        raw = input().split()
        valid = True
        for token in raw:
            if token == 'all':
                if not allow_all:
                    print("  Cannot use 'all' in this method. TRY AGAIN!")
                    valid = False; break
                group.extend(range(total_atoms))
            elif '-' in token:
                start, end = map(int, token.split('-'))
                group.extend(range(start - 1, end))
            elif token.isdigit():
                group.append(int(token) - 1)
            else:
                group.extend([j for j, lbl in enumerate(species) if lbl == token])
        if not valid:
            continue
        if group and all(0 <= idx < total_atoms for idx in group):
            return group
        print("  Wrong input atom-indexes! TRY AGAIN!")

def one_to_all(total_atoms,
               positions_cartesian,
               labels,
               image_offsets):
    
    while True:
        select = input(f"Choose the selected atom (  1 to {total_atoms:>3}): ")
        if select.isdigit() and 0 < int(select) <= total_atoms:
            index_select = int(select) - 1
            break
        print('WRONG No. of the selected atom')

    mask = np.arange(total_atoms) != index_select
    other_positions = positions_cartesian[mask]
    other_labels = [labels[i] for i in range(total_atoms) if i != index_select]
 
    min_distances = min_image_distances(positions_cartesian[index_select], other_positions, image_offsets)
    pair = [(labels[index_select], lbl) for lbl in other_labels]

    with open('distance-unsorted.dat', 'w') as o:
        o.write(f"# Distance between {labels[index_select]} and all other atoms\n")
        o.write("#   Atom1  Atom2     Distance\n")
        for (a1, a2), d in zip(pair, min_distances):
            o.write(f"  {a1:>5s}  {a2:>5s}  {d:>12.8f}\n")
        o.write(f"      Average   {np.mean(min_distances):>12.8f}\n")

    order = np.argsort(min_distances)
    with open('distance-sorted.dat', 'w') as o:
        o.write(f"# Distance between {labels[index_select]} and all other atoms\n")
        o.write("#   Atom1  Atom2     Distance\n")
        for i in order:
            a1, a2 = pair[i]
            o.write(f"  {a1:>5s}  {a2:>5s}  {min_distances[i]:>12.8f}\n")
        o.write(f"      Average   {np.mean(min_distances):>12.8f}\n")

def atom_pairs(total_atoms,
               positions_cartesian,
               labels,
               image_offsets):
    
    while True:
        inp = input("Enter number of pair atoms: ")
        if inp.isdigit() and int(inp) > 0:
            number_pair = int(inp); break
        print("Number of pair atoms must be a positive integer.")
 
    distances, pair = [], []
    for i in range(number_pair):
        print(f"For pair {i + 1:>3}")
        while True:
            s1 = input(f"  Choose the 1st selected atom of pair {i + 1:>3} (  1 to {total_atoms:>3}): ")
            if s1.isdigit() and 0 < int(s1) <= total_atoms:
                idx1 = int(s1) - 1; break
            print('WRONG No. of the 1st selected atom')
        while True:
            s2 = input(f"  Choose the 2nd selected atom of pair {i + 1:>3} (  1 to {total_atoms:>3}): ")
            if s2.isdigit() and 0 < int(s2) <= total_atoms:
                idx2 = int(s2) - 1; break
            print('WRONG No. of the 2nd selected atom')
 
        min_distance = min_image_distance(positions_cartesian[idx1], positions_cartesian[idx2], image_offsets)
        pair.append((labels[idx1], labels[idx2]))
        distances.append(min_distance)
 
    print("# Distance between 2 atoms")
    print("#   Atom1  Atom2     Distance")
    for (a1, a2), min_distance in zip(pair, distances):
        print(f"  {a1:>5s}  {a2:>5s}  {min_distance:>12.8f}")
    print(f"      Average   {np.mean(distances):>12.8f}")
 
    with open('distance-atom-atom.dat', 'w') as o:
        o.write("# Distance between 2 atoms\n")
        o.write("#   Atom1  Atom2     Distance\n")
        for (a1, a2), min_distance in zip(pair, distances):
            o.write(f"  {a1:>5s}  {a2:>5s}  {min_distance:>12.8f}\n")
        o.write(f"      Average   {np.mean(distances):>12.8f}\n")

def atom_molecule(total_atoms,
                  positions_cartesian,
                  species,
                  labels,
                  image_offsets):
    
    digits = len(str(total_atoms)) + 1
 
    while True:
        inp = input("Enter number of pair atom-molecule: ")
        if inp.isdigit() and int(inp) > 0:
            number_pair = int(inp); break
        print("Number of pair atom-molecule must be a positive integer.")
 
    distances, pair = [], []
    for i in range(number_pair):
        print(f"For pair {i + 1:>3}")
 
        while True:
            sel = input(f"  Choose the selected atom of pair {i + 1:>3} (  1 to {total_atoms:>3}): ")
            if sel.isdigit() and 0 < int(sel) <= total_atoms:
                index_select = int(sel) - 1; break
            print('WRONG No. of the selected atom')
 
        targets = parse_group(f"\nInput element-symbol and/or atom-indexes to choose ({1:>3} to {total_atoms:>3})\n"
"(Free-format input, e.g., 1 3 1-4 C H all)", total_atoms, species, allow_all=True)
 
        target_site = np.mean(positions_cartesian[targets], axis=0)
        min_distance = min_image_distance(positions_cartesian[index_select], target_site, image_offsets)
        pair.append((labels[index_select], str(i + 1).zfill(digits)))
        distances.append(min_distance)
 
    print("# Distance between selected atom and molecule")
    print("#   Atom   Molecule  Distance")
    for (atom, mol), min_distance in zip(pair, distances):
        print(f"  {atom:>5s}  {mol:>5s}  {min_distance:>12.8f}")
    print(f"      Average   {np.mean(distances):>12.8f}")
 
    with open('distance-atom-molecule.dat', 'w') as o:
        o.write("# Distance between selected atom and molecule\n")
        o.write("#   Atom   Molecule  Distance\n")
        for (atom, mol), min_distance in zip(pair, distances):
            o.write(f"  {atom:>5s}  {mol:>5s}  {min_distance:>12.8f}\n")
        o.write(f"      Average   {np.mean(distances):>12.8f}\n")

def z_distance(total_atoms,
               positions,
               species):
    
    print("Tip: this method can measure the thickness of your system.")
 
    # Substrate
    substrate_index = parse_group(f"\nSubstrate — input element-symbol and/or atom-indexes ({1:>3} to {total_atoms:>3})\n"
"(Free-format input, e.g., 1 3 1-4 C H  — 'all' not allowed)", total_atoms, species, allow_all=False)
 
    if len(substrate_index) == 1:
        highest_substrate = positions[substrate_index[0]]
    else:
        z_sub = positions[substrate_index, 2]
        top_candidates = [substrate_index[j] for j, z in enumerate(z_sub) if z == z_sub.max()]
        if len(top_candidates) == 1:
            highest_substrate = positions[top_candidates[0]]
        else:
            print(f"  The highest atoms in substrate : {[i + 1 for i in top_candidates]}")
            while True:
                sel = input(f"  Select atom in substrate (  1 to {total_atoms:>3}): ")
                if sel.isdigit() and int(sel) - 1 in top_candidates:
                    highest_substrate = positions[int(sel) - 1]; break
                print('WRONG No. of atom in substrate!')
 
    # Molecule
    adsorbent_index = parse_group(f"\nAdsorbent — input element-symbol and/or atom-indexes ({1:>3} to {total_atoms:>3})\n"
"(Free-format input, e.g., 1 3 1-4 C H  — 'all' not allowed)", total_atoms, species, allow_all=False)
 
    if len(adsorbent_index) == 1:
        lowest_adsorbent = positions[adsorbent_index[0]]
    else:
        z_ads = positions[adsorbent_index, 2]
        bot_candidates = [adsorbent_index[j] for j, z in enumerate(z_ads) if z == z_ads.min()]
        if len(bot_candidates) == 1:
            lowest_adsorbent = positions[bot_candidates[0]]
        else:
            print(f"  The lowest atoms in adsorbent : {[i + 1 for i in bot_candidates]}")
            while True:
                sel = input(f"  Select atom in adsorbent (  1 to {total_atoms:>3}): ")
                if sel.isdigit() and int(sel) - 1 in bot_candidates:
                    lowest_adsorbent = positions[int(sel) - 1]; break
                print('WRONG No. of atom in adsorbent!')
 
    distance = np.abs(lowest_adsorbent[2] - highest_substrate[2])
    print(f"Distance along z-axis is {distance:>12.8f} Angstrom.")

def main():
    if '-h' in argv or len(argv) != 2:
        usage()
 
    poscar = read_POSCAR(argv[1])
    mapping = mapping_elements(poscar["elements"],
                               poscar["atom_counts"],
                               poscar["positions_cartesian"],
                               poscar["positions_direct"],
                               poscar["species"],
                               poscar["selective_dynamics"],
                               poscar["flags"])
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])
    image_offsets = compute_image_offsets(poscar["lattice_matrix"])
 
    print("""
Choices of calculating distance
 1) Between selected atom and all other atoms
 2) Between 2 selected atoms
 3) Between selected atom and molecule
 4) Between highest atom in substrate and lowest atom in molecule (along z-axis only)""")
 
    while True:
        method = input("Enter choice : ")
        if method == '1':
            one_to_all(poscar["total_atoms"],
                       mapping["positions_cartesian"],
                       labels,
                       image_offsets)
            break
        elif method == '2':
            atom_pairs(poscar["total_atoms"],
                       mapping["positions_cartesian"],
                       labels,
                       image_offsets)
            break
        elif method == '3':
            atom_molecule(poscar["total_atoms"],
                          mapping["positions_cartesian"],
                          mapping["species"],
                          labels,
                          image_offsets)
            break
        elif method == '4':
            z_distance(poscar["total_atoms"],
                       mapping["positions_cartesian"],
                       mapping["species"])
            break
        else:
            print("ERROR! Wrong choice")

if __name__ == "__main__":
    main()
