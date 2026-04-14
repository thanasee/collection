#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
import multiprocessing as mp
from numba import njit


def usage():

    text = """
Usage: vaspTwist.py <input>

This script supports VASP5 structure file format (i.e. POSCAR)
for generating moiré twisted bilayer structures from a monolayer input file.
It searches for commensurate supercell vectors at candidate twist angles
and writes one POSCAR per valid moiré configuration.

Parameters are set interactively after reading the input file.

Only monolayer POSCAR files with vacuum space in the z-direction are supported.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

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
        raw_elements = lines[5].split()
        for name in raw_elements:
            elements.append(name.split('/')[0].split('_')[0])
        atom_counts = [int(x) for x in lines[6].split()]
        selective_dynamics = lines[7].lower().startswith('s')
        position_start = 9 if selective_dynamics else 8

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

    is_direct = lines[position_start - 1].strip().lower().startswith('d')
    if is_direct:
        positions_direct = positions % 1.0
        positions_cartesian = direct_to_cartesian(lattice_matrix, positions_direct)
    else:
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


def write_POSCAR(filepath,
                 title,
                 lattice_matrix,
                 elements,
                 atom_counts,
                 positions_direct,
                 selective_dynamics,
                 flags,
                 labels):

    with open(filepath, 'w') as o:
        o.write(f"{title}\n")
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
                        f"   {flag[0]:s}   {flag[1]:s}   {flag[2]:s}   {label:>6s}\n")
        else:
            for position, label in zip(positions_direct, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {label:>6s}\n")


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def direct_to_cartesian(lattice_matrix,
                         positions_direct):

    positions = positions_direct % 1.0
    positions_cartesian = np.dot(positions, lattice_matrix)

    return positions_cartesian


def cartesian_to_direct(lattice_matrix,
                         positions_cartesian):

    positions_direct = np.dot(positions_cartesian, np.linalg.inv(lattice_matrix)) % 1.0

    return positions_direct


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

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


def define_labels(elements,
                  atom_counts):

    digits = len(str(max(atom_counts))) + 1
    labels = [f"{symbol}{str(counter).zfill(digits)}"
              for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]

    return labels


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


# ---------------------------------------------------------------------------
# Moiré supercell search (Numba-accelerated)
# ---------------------------------------------------------------------------

@njit
def rotation_matrix(theta):

    rad = np.deg2rad(theta)
    return np.array([[np.cos(rad), -np.sin(rad), 0.0],
                     [np.sin(rad),  np.cos(rad), 0.0],
                     [0.0,          0.0,          1.0]], dtype=np.float64)


@njit
def angle_between_vectors(v1, v2):

    norm_v1 = np.sqrt(np.dot(v1, v1))
    norm_v2 = np.sqrt(np.dot(v2, v2))

    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)

    if cosine_angle < -1.0:
        cosine_angle = -1.0
    elif cosine_angle > 1.0:
        cosine_angle = 1.0

    return np.rad2deg(np.arccos(cosine_angle))


@njit
def find_moire_vectors_chunk(a1, a2,
                              b1, b2,
                              theta_chunk, combined_list,
                              max_strain):

    results = []
    for theta in theta_chunk:
        rot = rotation_matrix(theta)
        b1_rot = rot @ b1.T
        b2_rot = rot @ b2.T

        for n1 in combined_list:
            for n2 in combined_list:
                for m1 in combined_list:
                    for m2 in combined_list:
                        v_layer1 = n1 * a1 + n2 * a2
                        v_layer2 = m1 * b1_rot + m2 * b2_rot
                        norm_v1 = np.linalg.norm(v_layer1)
                        if norm_v1 == 0.0:
                            continue
                        strain = np.linalg.norm(v_layer1 - v_layer2) / norm_v1
                        if strain <= max_strain:
                            results.append((theta, n1, n2, strain, v_layer1))

    return results


def find_moire_vectors(a1, a2,
                        b1, b2,
                        theta_min, theta_max, theta_step,
                        n_min, n_max):

    theta_array = np.arange(theta_min, theta_max, theta_step)

    range_negative = list(range(-n_max, -n_min + 1))
    range_positive = list(range(n_min, n_max + 1))
    combined_list = np.array(range_negative + range_positive)

    num_cores = mp.cpu_count()
    chunk_size = max(1, len(theta_array) // num_cores)
    theta_chunks = [theta_array[i:i + chunk_size]
                    for i in range(0, len(theta_array), chunk_size)]

    with mp.Pool(processes=num_cores) as pool:
        chunk_results = pool.starmap(
            find_moire_vectors_chunk,
            [(a1, a2, b1, b2, chunk, combined_list, MAX_STRAIN)
             for chunk in theta_chunks]
        )

    moire_vectors = [item for sublist in chunk_results for item in sublist]

    return moire_vectors


# ---------------------------------------------------------------------------
# Supercell construction
# ---------------------------------------------------------------------------

def build_supercell(lattice_matrix,
                     positions_cartesian,
                     species,
                     selective_dynamics,
                     flags,
                     A1, A2):

    a1 = lattice_matrix[0]
    a2 = lattice_matrix[1]
    a3 = lattice_matrix[2]

    # Determine bounding box for replication
    n_max = int(np.ceil(max(np.linalg.norm(A1), np.linalg.norm(A2)) /
                        min(np.linalg.norm(a1), np.linalg.norm(a2)))) + 2

    supercell_positions_cartesian = []
    supercell_species = []
    supercell_flags = [] if selective_dynamics else None

    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            shift = n1 * a1 + n2 * a2
            for atom_index in range(len(positions_cartesian)):
                new_position = positions_cartesian[atom_index] + shift
                supercell_positions_cartesian.append(new_position)
                supercell_species.append(species[atom_index])
                if selective_dynamics:
                    supercell_flags.append(flags[atom_index])

    supercell_positions_cartesian = np.array(supercell_positions_cartesian)

    # New supercell lattice: A1, A2 in-plane, keep original c-axis
    new_lattice_matrix = np.array([A1, A2, a3])

    # Filter atoms inside the new supercell parallelogram
    new_positions_direct = np.dot(supercell_positions_cartesian,
                                  np.linalg.inv(new_lattice_matrix))
    inside_mask = (new_positions_direct[:, 0] >= -1e-8) & (new_positions_direct[:, 0] < 1.0 - 1e-8) & \
                  (new_positions_direct[:, 1] >= -1e-8) & (new_positions_direct[:, 1] < 1.0 - 1e-8)

    filtered_positions_direct = new_positions_direct[inside_mask] % 1.0
    filtered_species = [supercell_species[i] for i in range(len(supercell_species)) if inside_mask[i]]
    filtered_flags = None
    if selective_dynamics:
        filtered_flags = [supercell_flags[i] for i in range(len(supercell_flags)) if inside_mask[i]]
        filtered_flags = np.array(filtered_flags)

    return {"lattice_matrix": new_lattice_matrix,
            "positions_direct": filtered_positions_direct,
            "species": filtered_species,
            "flags": filtered_flags if selective_dynamics else None}


def build_twisted_bilayer(layer1, layer2_rotated,
                           theta,
                           selective_dynamics):

    # Stack layer2 above layer1 with 3 Å interlayer spacing
    layer1_cartesian = direct_to_cartesian(layer1["lattice_matrix"],
                                           layer1["positions_direct"])
    layer2_cartesian = direct_to_cartesian(layer2_rotated["lattice_matrix"],
                                           layer2_rotated["positions_direct"])

    z_max_layer1 = np.max(layer1_cartesian[:, 2])
    z_min_layer2 = np.min(layer2_cartesian[:, 2])
    interlayer_gap = 3.0
    z_shift = z_max_layer1 - z_min_layer2 + interlayer_gap

    layer2_cartesian[:, 2] += z_shift

    combined_cartesian = np.vstack((layer1_cartesian, layer2_cartesian))
    combined_species = list(layer1["species"]) + list(layer2_rotated["species"])

    combined_flags = None
    if selective_dynamics and layer1["flags"] is not None and layer2_rotated["flags"] is not None:
        combined_flags = np.vstack((layer1["flags"], layer2_rotated["flags"]))

    combined_positions_direct = cartesian_to_direct(layer1["lattice_matrix"],
                                                    combined_cartesian)
    combined_positions_direct = center_sheet(combined_positions_direct)

    return {"lattice_matrix": layer1["lattice_matrix"],
            "positions_direct": combined_positions_direct,
            "species": combined_species,
            "flags": combined_flags if selective_dynamics else None}


def collect_elements_and_counts(species,
                                 known_elements):

    elements_order = list(dict.fromkeys(
        e for e in known_elements for s in species if s == e
    ))
    atom_counts = [species.count(e) for e in elements_order]

    return elements_order, atom_counts


def build_candidates_for_theta(theta_key, vec_list,
                                monolayer_lattice_matrix,
                                monolayer_positions_cartesian,
                                monolayer_species,
                                monolayer_selective_dynamics,
                                monolayer_flags,
                                sort_elements,
                                known_elements):

    a1 = monolayer_lattice_matrix[0]
    a2 = monolayer_lattice_matrix[1]
    a3 = monolayer_lattice_matrix[2]
    theta = theta_key

    rot = rotation_matrix(theta)
    b1_rot = (rot @ a1.T)
    b2_rot = (rot @ a2.T)
    rotated_lattice_matrix = np.array([b1_rot, b2_rot, a3])

    theta_candidates = []
    seen_configurations = set()

    for i in range(len(vec_list)):
        strain_i, A1_vec = vec_list[i]
        for j in range(i + 1, len(vec_list)):
            strain_j, A2_vec = vec_list[j]

            cross_z = A1_vec[0] * A2_vec[1] - A1_vec[1] * A2_vec[0]
            if np.abs(cross_z) < 1e-6:
                continue

            config_key = (round(np.linalg.norm(A1_vec), 3),
                          round(np.linalg.norm(A2_vec), 3))
            if config_key in seen_configurations:
                continue
            seen_configurations.add(config_key)

            layer1 = build_supercell(monolayer_lattice_matrix,
                                      monolayer_positions_cartesian,
                                      monolayer_species,
                                      monolayer_selective_dynamics,
                                      monolayer_flags,
                                      A1_vec, A2_vec)

            layer2_rotated = build_supercell(rotated_lattice_matrix,
                                              monolayer_positions_cartesian,
                                              monolayer_species,
                                              monolayer_selective_dynamics,
                                              monolayer_flags,
                                              A1_vec, A2_vec)

            bilayer = build_twisted_bilayer(layer1, layer2_rotated,
                                             theta,
                                             monolayer_selective_dynamics)

            elements_order, atom_counts = collect_elements_and_counts(
                bilayer["species"], known_elements
            )

            bilayer_cartesian = direct_to_cartesian(bilayer["lattice_matrix"],
                                                    bilayer["positions_direct"])

            mapping = mapping_elements(elements_order,
                                        atom_counts,
                                        bilayer_cartesian,
                                        bilayer["positions_direct"],
                                        bilayer["species"],
                                        monolayer_selective_dynamics,
                                        bilayer["flags"],
                                        sort_elements)

            total_atoms_bilayer = sum(mapping["atom_counts"])
            if total_atoms_bilayer > MAX_ATOMS:
                continue

            max_strain_bilayer = max(strain_i, strain_j)
            theta_candidates.append({"theta": theta,
                                      "A1_vec": A1_vec,
                                      "A2_vec": A2_vec,
                                      "strain": max_strain_bilayer,
                                      "mapping": mapping,
                                      "bilayer_lattice_matrix": bilayer["lattice_matrix"],
                                      "total_atoms": total_atoms_bilayer})

    return theta_candidates


MAX_ATOMS  = 500
THETA_STEP = 0.1
MAX_STRAIN = 0.05
N_MIN      = 1
N_MAX      = 10


# ---------------------------------------------------------------------------
# Candidate selection prompt
# ---------------------------------------------------------------------------

def prompt_selection(candidates):

    print(f"\nFound {len(candidates)} candidate(s) with <= {MAX_ATOMS} atoms:\n")
    print(f"  {'No.':>4}  {'Theta (deg)':>12}  {'Total atoms':>11}  {'|A1| (Ang)':>10}  {'|A2| (Ang)':>10}  {'Strain (%)':>10}")
    print("  " + "-" * 68)
    for index, candidate in enumerate(candidates):
        theta       = candidate["theta"]
        total_atoms = candidate["total_atoms"]
        norm_A1     = np.linalg.norm(candidate["A1_vec"])
        norm_A2     = np.linalg.norm(candidate["A2_vec"])
        strain_pct  = candidate["strain"] * 100.0
        print(f"  {index + 1:>4}  {theta:>12.4f}  {total_atoms:>11}  {norm_A1:>10.4f}  {norm_A2:>10.4f}  {strain_pct:>9.2f}%")

    print()
    while True:
        raw = input("Enter candidate numbers to write (e.g. 1 3 5), or 'all': ").strip()
        if raw.lower() == 'all':
            return list(range(len(candidates)))
        try:
            chosen = [int(x) - 1 for x in raw.split()]
            if all(0 <= i < len(candidates) for i in chosen):
                return chosen
            print(f"ERROR! Numbers must be between 1 and {len(candidates)}.")
        except ValueError:
            print("ERROR! Enter integers separated by spaces, or 'all'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    if '-h' in argv or '--help' in argv or len(argv) != 2:
        usage()

    working_dir = os.getcwd()
    monolayer = read_POSCAR(argv[1])

    a1 = monolayer["lattice_matrix"][0]
    a2 = monolayer["lattice_matrix"][1]
    a3 = monolayer["lattice_matrix"][2]

    # For a twisted bilayer both layers share the same primitive lattice
    b1 = a1.copy()
    b2 = a2.copy()

    bilayer_elements = monolayer["elements"] + monolayer["elements"]
    sort_elements = check_elements(bilayer_elements)

    print(f"\nSearching for commensurate moiré vectors (0 to 180 deg, step = {THETA_STEP} deg)...")
    moire_vectors = find_moire_vectors(a1, a2,
                                        b1, b2,
                                        0.0, 180.0,
                                        THETA_STEP,
                                        N_MIN,
                                        N_MAX)

    if len(moire_vectors) == 0:
        print("No commensurate moiré vectors found with the given parameters.")
        exit(0)

    print(f"Found {len(moire_vectors)} raw result(s). Filtering candidates (<= {MAX_ATOMS} atoms)...\n")

    # Group strain-passing vectors by theta
    vectors_by_theta = {}
    for result in moire_vectors:
        theta  = result[0]
        strain = result[3]
        vec    = np.array(result[4])
        vectors_by_theta.setdefault(round(theta, 4), []).append((strain, vec))

    known_elements = sort_elements if sort_elements is not None else monolayer["elements"]

    num_cores = mp.cpu_count()
    with mp.Pool(processes=num_cores) as pool:
        per_theta_results = pool.starmap(
            build_candidates_for_theta,
            [(theta_key, vec_list,
              monolayer["lattice_matrix"],
              monolayer["positions_cartesian"],
              monolayer["species"],
              monolayer["selective_dynamics"],
              monolayer["flags"],
              sort_elements,
              known_elements)
             for theta_key, vec_list in vectors_by_theta.items()]
        )

    candidates = [c for theta_list in per_theta_results for c in theta_list]

    # Keep only the candidate with fewest atoms for each theta
    best_per_theta = {}
    for c in candidates:
        theta_key = c["theta"]
        if theta_key not in best_per_theta or c["total_atoms"] < best_per_theta[theta_key]["total_atoms"]:
            best_per_theta[theta_key] = c
    candidates = list(best_per_theta.values())

    if len(candidates) == 0:
        print(f"No candidates found with <= {MAX_ATOMS} atoms. Try relaxing the parameters.")
        exit(0)

    candidates.sort(key=lambda c: (c["total_atoms"], c["theta"]))

    chosen_indices = prompt_selection(candidates)

    written_count = 0
    for index in chosen_indices:
        candidate     = candidates[index]
        theta         = candidate["theta"]
        mapping       = candidate["mapping"]
        lattice_matrix = candidate["bilayer_lattice_matrix"]
        total_atoms   = candidate["total_atoms"]

        labels = define_labels(mapping["elements"], mapping["atom_counts"])

        output_dir = os.path.join(working_dir, f"twist_{theta:.4f}deg_{total_atoms}atoms")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "POSCAR")

        title = f"Generated by vaspTwist.py | theta = {theta:.4f} deg | {total_atoms} atoms | strain = {candidate['strain'] * 100:.2f}%"
        write_POSCAR(output_path,
                     title,
                     lattice_matrix,
                     mapping["elements"],
                     mapping["atom_counts"],
                     mapping["positions_direct"],
                     monolayer["selective_dynamics"],
                     mapping["flags"],
                     labels)

        print(f"  Written: {output_dir}/POSCAR"
              f"  (theta = {theta:.4f} deg, {total_atoms} atoms)")
        written_count += 1

    print(f"\nFinished! Written {written_count} POSCAR(s).\n")


if __name__ == "__main__":
    main()
