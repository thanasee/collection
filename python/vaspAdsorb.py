#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: vaspAdsorb.py <substrate input> <adsorbent input> <output>

This script design to prepare position file for adsorption calculation
which has 2 main functions:ฆ
  1) adsorb on specific site
  2) adsorb around target atom
This script support VASP5 position file format (i.e. POSCAR).

This script was inspired by Aroon Ananchunsook
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
        o.write("Generated by vaspAdsorb.py code\n")
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

def selection_atoms(prompt,
                    total_atoms,
                    species):
    
    print(prompt)
    while True:
        selected = []
        for token in input().split():
            if token == 'all':
                selected.extend(range(total_atoms))
                break
            elif '-' in token and not token.lstrip('-').isdigit():
                start, end = map(int, token.split('-'))
                selected.extend(range(start - 1, end))
            elif token.isnumeric():
                selected.append(int(token) - 1)
            else:
                selected.extend(i for i, s in enumerate(species) if s == token)
        if selected and all(0 <= idx < total_atoms for idx in selected):
            return selected
        print("Wrong input atom-indexes! TRY AGAIN!")

def input_direct(lattice_matrix):
    
    coords = []
    for direction in ('a', 'b'):
        while True:
            val = input(f"Enter position in {direction} direction (direct): ")
            if val.replace('.', '', 1).isdigit():
                coords.append(float(val))
                break
            print("Wrong input! Try again")
    coords.append(0.0)

    return np.dot(coords, lattice_matrix)

def place_ontop(lattice_matrix_substrate,
                total_atoms_substrate,
                total_atoms_adsorbent,
                positions_substrate,
                positions_adsorbent,
                species_substrate,
                species_adsorbent,
                selective_dynamics,
                flags_adsorbent,
                number_adsorbent,
                delta):

    print("""
Method of reference height of substrate
1) highest atom in substrate
2) selected atom in substrate
3) average height from atoms in substrate""")
    while True:
        option_height = input("Enter choice: ")
        if option_height == '1':
            z_substrate = np.max(positions_substrate[:, 2])
            break
        elif option_height == '2':
            while True:
                select_substrate = input(f"Select atom in substrate ({1:>3} to {total_atoms_substrate:>3}): ")
                if select_substrate.isdigit() and 0 <= int(select_substrate) - 1 < total_atoms_substrate:
                    break
                print("WRONG No. of atom in substrate!")
            z_substrate = positions_substrate[int(select_substrate) - 1, 2]
            break
        elif option_height == '3':
            z_substrate = np.mean(positions_substrate[:, 2])
            break
        else:
            print("ERROR!! Choose again")

    # Reference height of adsorbent (lowest atom)
    reference_adsorbent = np.zeros(3)
    reference_adsorbent[2] = np.min(positions_adsorbent[:, 2])

    # Set distance in z component
    distance = np.zeros(3)
    distance[2] = z_substrate - reference_adsorbent[2] + delta

    # Choose reference xy position of adsorbent
    if total_atoms_adsorbent == 1:
        reference_adsorbent[:2] = positions_adsorbent[0, :2]
    else:
        print("""
Choices of selecting the drop point of adsorbent
1) center of adsorbent
2) selected atom in adsorbent or lowest atom in adsorbent""")
        while True:
            option_point = input("Enter choice: ")
            if option_point == '1':
                reference_adsorbent[:2] = np.mean(positions_adsorbent, axis=0)[:2]
                break
            elif option_point == '2':
                lowest = [i + 1 for i in range(total_atoms_adsorbent) if np.isclose(positions_adsorbent[i, 2], np.min(positions_adsorbent[:, 2]))]
                if len(lowest) == 1:
                    select_adsorbent = lowest[0] - 1
                else:
                    print(f"The lowest atom in adsorbent : {lowest}")
                    for j in range(total_atoms_adsorbent):
                        print(f"{species_adsorbent[j]} atom : {j + 1:>3}")
                    while True:
                        select_adsorbent = input(f"Select atom in adsorbent (  1 to {total_atoms_adsorbent:>3}): ")
                        if select_adsorbent.isdigit() and 0 <= int(select_adsorbent) - 1 < total_atoms_adsorbent:
                            select_adsorbent = int(select_adsorbent) - 1
                            break
                        print('WRONG No. of atom in adsorbent!')
                reference_adsorbent = np.copy(positions_adsorbent[select_adsorbent])
                break
            else:
                print("ERROR!! Choose again")

    # Place each adsorbent copy
    new_positions_adsorbent = []
    new_species_adsorbent = []
    new_flags_adsorbent = [] if selective_dynamics else None

    for n in range(number_adsorbent):
        print(f"""
Choices of positioning adsorbent for adsorbent {n+1:>2}
1) Choose atoms surround the positioning point
   If 1 atom means ontop that atom
   If 2 or more atoms mean on top of center point of these atoms
2) Custom position in Direct coordinate""")
        while True:
            option_position = input("Enter choice: ")
            if option_position == '1':
                prompt = (f"\nInput element-symbol and/or atom-indexes to choose "
                          f"({1:>3} to {total_atoms_substrate:>3})\n"
                          f"(Free-format input, e.g., 1 3 1-4 C H all)")
                selected_atoms = selection_atoms(prompt, total_atoms_substrate, species_substrate)
                target = np.mean(positions_substrate[selected_atoms], axis=0)
                break
            elif option_position == '2':
                target = input_direct(lattice_matrix_substrate)
                break
            else:
                print("ERROR!! Choose again")

        distance[:2] = target[:2] - reference_adsorbent[:2]
        new_positions_adsorbent.append(positions_adsorbent + distance)
        new_species_adsorbent.append(species_adsorbent)
        if selective_dynamics:
            new_flags_adsorbent.append(flags_adsorbent)

    return {"positions_adsorbent": np.vstack(new_positions_adsorbent),
            "species_adsorbent": np.vstack(new_species_adsorbent),
            "flags_adsorbent": np.vstack(new_flags_adsorbent) if selective_dynamics else None}

def place_around(lattice_matrix_substrate,
                 total_atoms_substrate,
                 positions_substrate,
                 positions_adsorbent,
                 species_substrate,
                 species_adsorbent,
                 selective_dynamics,
                 flags_adsorbent,
                 number_adsorbent,
                 delta):

    reference_adsorbent = np.zeros(3)
    reference_adsorbent[:2] = np.mean(positions_adsorbent, axis=0)[:2]
    reference_adsorbent[2] = np.min(positions_adsorbent[:, 2])

    # Choose target atom in substrate to decorate around
    while True:
        target_atom = input(f"Select target atom in substrate ({1:>3} to {total_atoms_substrate:>3}): ")
        if target_atom.isdigit() and 0 <= int(target_atom) - 1 < total_atoms_substrate:
            break
        print("WRONG No. of atom in substrate!")
    target_center = positions_substrate[int(target_atom) - 1, :]

    # Choose initial adsorption site direction
    print("""
Choices of define initial adsorption site
1) Choose atoms surround the positioning point
   If 1 atom means ontop that atom
   If 2 or more atoms mean on top of center point of these atoms
2) Custom position in Direct coordinate""")
    while True:
        option_site = input("Enter choice: ")
        if option_site == '1':
            prompt = (f"\nInput element-symbol and/or atom-indexes to choose "
                      f"({1:>3} to {total_atoms_substrate:>3})\n"
                      f"(Free-format input, e.g., 1 3 1-4 C H all)")
            targets = selection_atoms(prompt, total_atoms_substrate, species_substrate)
            target_site = np.mean(positions_substrate[targets], axis=0)
            break
        elif option_site == '2':
            target_site = input_direct(lattice_matrix_substrate)
            break
        else:
            print("ERROR!! Choose again")

    # Compute initial displacement direction
    xy_distance = target_site[:2] - target_center[:2]
    norm = np.linalg.norm(xy_distance)
    if norm == 0:
        print("ERROR! Initial site cannot be the same as the target atom.")
        exit(1)
    xy_unit = xy_distance / norm

    distance = np.zeros(3)
    distance[:2] = delta * xy_unit
    distance[2] = target_center[2] - reference_adsorbent[2]

    angle_step = 2 * np.pi / number_adsorbent

    new_positions_adsorbent = []
    new_species_adsorbent = []
    new_flags_adsorbent = [] if selective_dynamics else None

    for i in range(number_adsorbent):
        rotate_matrix = rotation_matrix(i, angle_step)

        displacement = np.dot(rotate_matrix, distance)
        new_site = target_center + displacement

        rotate_position_adsorbent = (np.dot(positions_adsorbent - reference_adsorbent,
                                            rotate_matrix.T) + reference_adsorbent)
        translate = new_site - reference_adsorbent

        new_positions_adsorbent.append(rotate_position_adsorbent + translate)
        new_species_adsorbent.append(species_adsorbent)
        if selective_dynamics:
            new_flags_adsorbent.append(flags_adsorbent)

    return {"positions_adsorbent": np.vstack(new_positions_adsorbent),
            "species_adsorbent": np.vstack(new_species_adsorbent),
            "flags_adsorbent": np.vstack(new_flags_adsorbent) if selective_dynamics else None}

def rotation_matrix(i,
                    angle_step):
    
    degree = i * angle_step

    # define trigonometry functions
    sin = np.sin(degree)
    cos = np.cos(degree)
    u = np.array([0., 0., 1.])

    # Matrix of rotation
    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)
    
    return rotate

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

def main():

    if os.environ.get('USER') == 'nchotsis':
        print("If you're so unhappy with how I refine my code, feel free to write it yourself.")
    
    if '-h' in argv or '--help' in argv or len(argv) != 4:
        usage()
    
    substrate = read_POSCAR(argv[1])
    adsorbent = read_POSCAR(argv[2])
    selective_dynamics_substrate = substrate["selective_dynamics"]
    selective_dynamics_adsorbent = adsorbent["selective_dynamics"]
    selective_dynamics = selective_dynamics_substrate or selective_dynamics_adsorbent
    flags_substrate = substrate["flags"]
    flags_adsorbent = adsorbent["flags"]
    if selective_dynamics:
        if selective_dynamics_substrate and not selective_dynamics_adsorbent:
            flags_adsorbent = np.full((adsorbent["total_atoms"], 3), 'T')
        elif selective_dynamics_adsorbent and not selective_dynamics_substrate:
            flags_substrate = np.full((substrate["total_atoms"], 3), 'T')
    while True:
        input_number = input("Enter number of adsorbent: ")
        if input_number.isdigit() and int(input_number) > 0:
            number_adsorbent = int(input_number)
            break
        print("Number of adsorbent must be positive integer")
    while True:
        delta = input("Enter distance between substrate and adsorbent (Angstrom): ")
        if delta.lstrip('-').replace('.', '', 1).isdigit():
            delta = float(delta)
            break
        print("Distance must be number")
    new_total_atoms_adsorbent = adsorbent["total_atoms"] * number_adsorbent
    new_atom_counts_adsorbent = adsorbent["atom_counts"] * number_adsorbent
    total_atoms = substrate["total_atoms"] + new_total_atoms_adsorbent
    atom_counts = substrate["atom_counts"] + new_atom_counts_adsorbent
    elements = substrate["elements"] + adsorbent["elements"]
    species = substrate["species"] + adsorbent["species"] * number_adsorbent
    print("""
Method of positioning adsorbent
1) On top specific site
2) Around target atom""")
    while True:
        option = input("Enter choice: ")
        if option == '1':
            place = place_ontop(substrate["lattice_matrix"],
                                substrate["total_atoms"],
                                adsorbent["total_atoms"],
                                substrate["positions_cartesian"],
                                adsorbent["positions_cartesian"],
                                substrate["species"],
                                adsorbent["species"],
                                selective_dynamics,
                                flags_adsorbent,
                                number_adsorbent,
                                delta)
            break
        elif option == '2':
            place = place_around(substrate["lattice_matrix"],
                                 substrate["total_atoms"],
                                 substrate["positions_cartesian"],
                                 adsorbent["positions_cartesian"],
                                 substrate["species"],
                                 adsorbent["species"],
                                 selective_dynamics,
                                 flags_adsorbent,
                                 number_adsorbent,
                                 delta)
            break
        else:
            print("ERROR! Wrong choice")
    new_positions_cartesian = np.vstack((substrate["positions_cartesian"], place["positions_adsorbent"]))
    new_species = np.vstack((substrate["species"], place["species_adsorbent"]))
    if selective_dynamics:
        new_flags = np.vstack((flags_substrate, place["flags_adsorbent"]))
    else:
        # Prompt user for optional selective dynamics
        while True:
            selective = input("Want to selective dynamic? (Y/N): ").strip().upper()
            if selective[0] == 'Y':
                selective_dynamics = True
                new_flags = np.full((total_atoms, 3), 'T')

                fix_prompt = (f"\nInput element-symbol and/or atom-indexes to choose "
f"(  1 to {total_atoms:>3})\n"
f"(Free-format input, e.g., 1 3 1-4 C H all)")
                fixed_atoms = selection_atoms(fix_prompt, total_atoms, species)
                fix_coordinates = select_direction()
                
                for atom in fixed_atoms:
                    for direction in fix_coordinates:
                        new_flags[atom][direction] = 'F'
                break
            elif selective[0] == 'N':
                new_flags = None
                break
            else:
                print("ERROR! Wrong input selective dynamic mode")
    new_positions_direct = cartesian_to_direct(substrate["lattice_matrix"],
                                               new_positions_cartesian)
    mapping = mapping_elements(elements,
                               atom_counts,
                               new_positions_cartesian,
                               new_positions_direct,
                               new_species,
                               selective_dynamics,
                               new_flags)
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])
    write_POSCAR(argv[3],
                 substrate["lattice_matrix"],
                 mapping["elements"],
                 mapping["atom_counts"],
                 mapping["positions_direct"],
                 selective_dynamics,
                 mapping["flags"],
                 labels)
    
    print(f"\nAdsorption structure written to: {argv[3]}")
    print("-" * 49)
    print("  Element  |  Substrate  |  Adsorbent  |  Total")
    print("-" * 49)
    for element, count in zip(mapping["elements"], mapping["atom_counts"]):
        n_sub = sum(1 for s in substrate["species"] if s == element)
        n_ads = count - n_sub
        print(f"  {element:<9}|  {n_sub:<11}|  {n_ads:<11}|  {count}")
    print("-" * 49)
    print(f"  Total    |  {substrate['total_atoms']:<11}|  {new_total_atoms_adsorbent:<11}|  {total_atoms}")
    print("-" * 49 + "\n")

if __name__ == "__main__":
    main()
