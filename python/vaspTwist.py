#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

MAX_ATOMS   = 1000
MAX_SC      = 50
STRAIN_SHOW = 5.0

def usage():
    
    text = f"""
Usage: vaspTwist.py <input>
 
Generate twisted bilayer POSCAR files from a 2D monolayer POSCAR.
 
The script automatically:
  1. Searches all (m, n) integer pairs until the supercell exceeds
     {MAX_ATOMS} atoms (both layers combined).
  2. Computes the exact twist angle and strain for each pair.
  3. Shows a table of solutions with strain <= {STRAIN_SHOW}%.
  4. Asks which angle(s) to build.
  5. Writes one POSCAR per chosen angle (AA stacking, no in-plane shift).
 
Strain definition:
  For each (m, n), independent supercells are built from the original
  (bottom) and rotated (top) lattices.  Strain = max relative mismatch
  across supercell vector lengths and internal angle between the two cells.
 
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

def write_POSCAR(filepath,
                 lattice_matrix,
                 elements,
                 atom_counts,
                 positions_direct,
                 selective_dynamics,
                 flags,
                 labels):
    
    with open(filepath, 'w') as o:
        o.write("Generated by vaspStack.py code\n")
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

def gcd(a, b):
    a, b = abs(int(round(a))), abs(int(round(b)))
    while b:
        a, b = b, a % b
    return a

def input_expansion(supercell_matrix):
    
    expansion_matrix = np.eye(3, dtype=float)
    expansion_matrix[:2, :2] = supercell_matrix.astype(float)
    
    det = np.linalg.det(expansion_matrix)
    det_int = int(round(det))
    
    if det <= 1e-10:
        print(f"Invalid expansion matrix: determinant = {det:.6f} (must be a positive integer).")
        exit(0)
    if abs(det - det_int) >= 1e-6 or det_int <= 0:
        print(f"Invalid expansion matrix: determinant = {det:.6f} (must be a positive integer).")
        exit(0)
    
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

def unwrap(positions_direct):
    
    reference = np.copy(positions_direct[0])
    delta = positions_direct - reference
    delta -= np.round(delta)
    
    return reference, reference + delta

def center_sheet(positions_direct):
    
    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    vacuum = 2
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5
    
    return new % 1.0

def build_rotate(degrees):
    
    axis = 2
    u = np.array([1. if i == axis else 0. for i in range(3)])
    radians = np.radians(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)
    
    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)
    
    return rotate

def cutoff_radius(m,
                  n,
                  lattice_matrix):
    
    max_length = np.max(np.linalg.norm(lattice_matrix[0]), np.linalg.norm(lattice_matrix[1]))
    area = np.linalg.norm(np.cross(lattice_matrix[0], lattice_matrix[1]))
    
    max_first_length = (m + n) * max_length
    
    min_second_length = area / max_length
    if min_second_length < 1e-10 or max_length < 1e-10:
        return m + n + 2
    
    radians = int(np.ceil(max_first_length / min_second_length)) + 2
    
    return radians

def find_twist_vector(m,
                      n,
                      lattice_matrix):
    
    first_vector = lattice_matrix[0] * m + lattice_matrix[1] * n
    first_length = np.linalg.norm(first_vector)
    first_area = np.linalg.norm(np.cross(lattice_matrix[0], lattice_matrix[1]))
    radians = cutoff_radius(m, n, lattice_matrix)
    
    best_diff = np.inf
    second_vector = None
    best_number = 0
    
    for p in range(-radians, radians + 1):
        for q in range(-radians, radians + 1):
            if p == 0 and q == 0:
                continue
            candidate_vector = lattice_matrix[0] * p + lattice_matrix[1] * q
            candidate_length = np.linalg.norm(candidate_vector)
            if candidate_length < 1e-10:
                continue
            sin = np.linalg.norm(np.cross(first_vector, candidate_vector)) / (first_length * candidate_length)
            if sin < 1e-6:
                continue
            diff = np.abs(candidate_length - first_length) / first_length
            if diff < best_diff:
                best_diff = diff
                second_vector = candidate_vector
                best_number = max(1, round(np.linalg.norm(np.cross(first_vector, candidate_vector)) / first_area))
    
    return first_vector, second_vector, best_number

def vector_angle(first_vector,
                 second_vector):
    
    gamma = np.degrees(np.arccos(np.clip((np.dot(first_vector, second_vector) / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))), -1, 1)))

    return gamma

def detect_strain(bottom_first_vector,
                  bottom_second_vector,
                  top_first_vector,
                  top_second_vector):
    
    bottom_first_length, bottom_second_length = np.linalg.norm(bottom_first_vector), np.linalg.norm(bottom_second_vector)
    top_first_length, top_second_length = np.linalg.norm(top_first_vector), np.linalg.norm(top_second_vector)
    
    if np.min(bottom_first_length, bottom_second_length, top_first_length, top_second_length) < 1e-10:
        return 100., 100., 100., 100.
    
    first_strain = np.abs(bottom_first_length - top_first_length) / np.max(bottom_first_length, top_first_length) * 100.
    second_strain = np.abs(bottom_second_length - top_second_length) / np.max(bottom_second_length, top_second_length) * 100.
    gamma_strain = np.abs(vector_angle(bottom_first_vector, bottom_second_vector) - vector_angle(top_first_vector, top_second_vector)) / 180. * 100.
    
    return np.max(first_strain, second_strain, gamma_strain), first_strain, second_strain, gamma_strain

def collect_twist_angle(lattice_matrix,
                        total_atoms):
    
    area = np.linalg.norm(np.cross(lattice_matrix[0], lattice_matrix[1]))
    
    solutions = []
    seen_gamma = {}
    
    for m in range(1, MAX_ATOMS + 1):
        if 2 * m * total_atoms > MAX_ATOMS:
            break
        
        for n in range(0, m + 1):
            if gcd(m, n) != 1:
                continue
            
            bottom_first_vector, bottom_second_vector, best_number = find_twist_vector(m,
                                                                                       n,
                                                                                       lattice_matrix)
            if bottom_second_vector is None or best_number == 0:
                continue
            if np.linalg.norm(bottom_second_vector) < 1e-10:
                continue
            
            new_total_atoms = 2 * best_number * total_atoms
            if new_total_atoms > MAX_ATOMS:
                continue
            
            gamma = vector_angle(bottom_first_vector,
                                 bottom_second_vector)
            if gamma < 1e-4 or gamma > 180. - 1e-4:
                continue
            
            rotate_matrix = build_rotate(gamma)
            top_lattice_matrix = np.dot(rotate_matrix, lattice_matrix.T).T
            top_first_vector, top_second_vector, _ = find_twist_vector(m,
                                                                       n,
                                                                       top_lattice_matrix)
            if top_second_vector is None:
                continue
            
            strain, first_strain, second_strain, gamma_strain = detect_strain(bottom_first_vector,
                                                                              bottom_second_vector,
                                                                              top_first_vector,
                                                                              top_second_vector)
            
            new_lattice_matrix = lattice_matrix.copy()
            new_lattice_matrix[0] = (top_first_vector + bottom_first_vector) / 2
            new_lattice_matrix[1] = (top_second_vector + bottom_second_vector) / 2
            new_area = np.linalg.norm(np.cross(new_lattice_matrix[0], new_lattice_matrix[1]))
            ratio = new_area / area
            
            M_bottom = np.array([np.round(np.dot(bottom_first_vector, np.linalg.inv(lattice_matrix))).astype(int),
                                 np.round(np.dot(bottom_second_vector, np.linalg.inv(lattice_matrix))).astype(int)])
            M_top = np.array([np.round(np.dot(top_first_vector, np.linalg.inv(top_lattice_matrix))).astype(int),
                                 np.round(np.dot(top_second_vector, np.linalg.inv(top_lattice_matrix))).astype(int)])
            
            if np.linalg.det(M_bottom) < 0.5 or np.linalg.det(M_top) < 0.5:
                continue
            
            gamma_key = round(gamma, 3)
            data = {"gamma": gamma,
                    "m": m,
                    "n": n,
                    "best_number": best_number,
                    "M_bottom": M_bottom,
                    "M_top": M_top,
                    "lattice_matrix": new_lattice_matrix,
                    "strain": strain,
                    "first_strain": first_strain,
                    "second_strain": second_strain,
                    "gamma_strain": gamma_strain,
                    "ratio": ratio,
                    "total_atoms": new_total_atoms}
            
            if gamma_key in seen_gamma:
                idx = seen_gamma[gamma_key]
                if strain < solutions[idx]["strain"]:
                    solutions[idx] = data
            else:
                seen_gamma[gamma_key] = len(solutions)
                solutions.append(data)
    
    solutions.sort(key=lambda x: x["gamma"])
    
    return solutions

def build_bilayer(atom_counts,
                  first_positions_direct,
                  second_positions_direct,
                  species,
                  selective_dynamics,
                  flags):
    
    new_atom_counts = atom_counts + atom_counts
    new_positions_direct = np.vstack((first_positions_direct, second_positions_direct))
    center_positions_direct = center_sheet(new_positions_direct)
    new_species = list(species) + list(species)
    
    new_flags = None
    if selective_dynamics:
        new_flags = np.vstack((flags, flags))
    
    return {"atom_counts": new_atom_counts,
            "positions_direct": center_positions_direct,
            "species": new_species,
            "flags": new_flags if selective_dynamics else None}

def main():
    
    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()

    working_dir = os.getcwd()
    monolayer = read_POSCAR(argv[1])
    first_positions_direct = center_sheet(monolayer["positions_direct"])
    thickness, second_positions_cartesion = build_second_layer(monolayer["positions_cartesian"])
    if thickness > 1e-8:
        while True:
            flip = input("Want to flip the second later (Y/N): ").strip().upper()
            if flip[0] == 'Y':
                second_positions_cartesion = mirror_image(second_positions_cartesion)
                break
            elif flip[0] == 'N':
                break
            else:
                print("ERROR! Yes or No only.")
    
    second_positions_direct = cartesian_to_direct(monolayer["lattice_matrix"],
                                                  second_positions_cartesion)
    bilayer_elements = monolayer["elements"] + monolayer["elements"]
    sort_elements = check_elements(bilayer_elements)
    
    for i in range(10):
        for j in range(10):
            shift_a = i / 10.
            shift_b = j / 10.
            shift_positions_direct = shift_sheet(second_positions_direct,
                                                 shift_a,
                                                 shift_b)
            bilayer = build_bilayer(monolayer["atom_counts"],
                                    first_positions_direct,
                                    shift_positions_direct,
                                    monolayer["species"],
                                    monolayer["selective_dynamics"],
                                    monolayer["flags"])
            bilayer_positions_cartesion = direct_to_cartesian(monolayer["lattice_matrix"],
                                                              bilayer["positions_direct"])
            mapping = mapping_elements(bilayer_elements,
                                       bilayer["atom_counts"],
                                       bilayer_positions_cartesion,
                                       bilayer["positions_direct"],
                                       bilayer["species"],
                                       monolayer["selective_dynamics"],
                                       bilayer["flags"],
                                       sort_elements)
            labels = define_labels(mapping["elements"],
                                   mapping["atom_counts"])
            output_dir = os.path.join(working_dir, f"stack_a{shift_a:.1f}_b{shift_b:.1f}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "POSCAR")
            write_POSCAR(output_path,
                         monolayer["lattice_matrix"],
                         mapping["elements"],
                         mapping["atom_counts"],
                         mapping["positions_direct"],
                         monolayer["selective_dynamics"],
                         mapping["flags"],
                         labels)
    print("Written 100 POSCARs Finished!\n")

if __name__ == "__main__":
    main()
