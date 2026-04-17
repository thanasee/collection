#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
import multiprocessing as mp
from numba import njit


def usage():
    """Print usage information and exit."""
    text = """
Usage:
  vaspTwist.py <bottom>         Homobilayer  — twist one monolayer against itself
  vaspTwist.py <bottom> <top>   Heterobilayer — twist two different monolayers

This script supports VASP5 structure file format (i.e. POSCAR)
for generating moiré twisted bilayer structures from monolayer input files.
It searches for commensurate supercell vectors at candidate twist angles
and writes one POSCAR per stacking configuration.

Only monolayer POSCAR files with vacuum space in the z-direction are supported.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

MAX_ATOMS  = 200    # Maximum number of atoms in the bilayer supercell
THETA_STEP = 0.1    # Twist angle search step (degrees)
MAX_STRAIN = 0.05   # Maximum symmetric relative distance for vector coincidence
N_MIN      = 1      # Minimum supercell index
N_MAX      = 10     # Maximum supercell index

def read_POSCAR(filepath):
    """Read a VASP POSCAR file and return its contents as a dictionary.

    Supports both VASP4 (no element line) and VASP5 (with element line) formats,
    scalar and negative (volume-based) scaling factors, a 3-component scaling
    vector, Selective Dynamics, and both Direct and Cartesian coordinate modes.

    Parameters
    ----------
    filepath : str
        Path to the POSCAR file to read.

    Returns
    -------
    dict with keys:
        lattice_matrix      : np.ndarray, shape (3, 3)  — lattice vectors in Å
        elements            : list[str]                 — element symbols
        atom_counts         : list[int]                 — number of atoms per element
        total_atoms         : int                       — total number of atoms
        positions_cartesian : np.ndarray, shape (N, 3)  — Cartesian coordinates in Å
        positions_direct    : np.ndarray, shape (N, 3)  — fractional coordinates
        species             : list[str]                 — element symbol per atom
        selective_dynamics  : bool                      — whether Selective Dynamics is present
        flags               : np.ndarray or None        — T/F flags per atom, or None
    """

    if not os.path.exists(filepath):
        print(f"ERROR!\nFile: {filepath} does not exist.")
        exit(1)

    with open(filepath, 'r') as poscar:
        lines = poscar.readlines()

    # Parse the scaling factor (line 2):
    # - 1 value  : uniform scalar; negative means target volume in Å**3
    # - 3 values : per-axis scale applied row-wise to the lattice matrix
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

    # Detect VASP4 vs VASP5 format by checking whether line 6 starts with a number.
    # VASP4 has no element-symbol line, so the user is prompted for species names.
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
        # VASP5 format: element symbols present.
        # Strip potential PAW/GGA suffixes such as '_pv' or '/GGA'.
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

    # Build a per-atom species list (e.g. ['Mo', 'Mo', 'S', 'S', 'S'])
    species = [x for i, x in enumerate(elements)
               for _ in range(atom_counts[i])]

    # Read Selective Dynamics T/F flags if present
    flags = None
    if selective_dynamics:
        flags = np.array([[x for x in lines[i].split()[3:6]]
                          for i in range(position_start, position_stop)])

    # Convert coordinates to both Direct and Cartesian representations
    is_direct = lines[position_start - 1].strip().lower().startswith('d')
    if is_direct:
        positions_direct = positions % 1.0
        positions_cartesian = direct_to_cartesian(lattice_matrix, positions_direct)
    else:
        positions_cartesian = positions * scale
        positions_direct = cartesian_to_direct(lattice_matrix, positions_cartesian)

    return {"lattice_matrix":     lattice_matrix,
            "elements":           elements,
            "atom_counts":        atom_counts,
            "total_atoms":        total_atoms,
            "positions_cartesian": positions_cartesian,
            "positions_direct":   positions_direct,
            "species":            species,
            "selective_dynamics": selective_dynamics,
            "flags":              flags if selective_dynamics else None}


def direct_to_cartesian(lattice_matrix, positions_direct):
    """Convert fractional (Direct) coordinates to Cartesian coordinates.

    Uses the relation:  r_cart = r_direct @ lattice_matrix

    Parameters
    ----------
    lattice_matrix    : np.ndarray, shape (3, 3) — row vectors of the lattice in Å
    positions_direct  : np.ndarray, shape (N, 3) — fractional coordinates

    Returns
    -------
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å
    """

    positions = positions_direct % 1.0
    positions_cartesian = np.dot(positions, lattice_matrix)

    return positions_cartesian


def cartesian_to_direct(lattice_matrix, positions_cartesian):
    """Convert Cartesian coordinates to fractional (Direct) coordinates.

    Uses the relation:  r_direct = r_cart @ lattice_matrix⁻¹

    Parameters
    ----------
    lattice_matrix      : np.ndarray, shape (3, 3) — row vectors of the lattice in Å
    positions_cartesian : np.ndarray, shape (N, 3) — Cartesian coordinates in Å

    Returns
    -------
    positions_direct : np.ndarray, shape (N, 3) — fractional coordinates in [0, 1)
    """

    positions_direct = np.dot(positions_cartesian, np.linalg.inv(lattice_matrix)) % 1.0

    return positions_direct


def check_elements(elements):
    """Check for duplicate element symbols and prompt the user for a canonical order.

    If duplicate symbols are found (e.g. ['Mo', 'S', 'Mo']), the user is asked
    to specify the desired ordering of the unique species. An empty input accepts
    the default order (first-occurrence order).

    Parameters
    ----------
    elements : list[str] — element symbols as parsed from the POSCAR

    Returns
    -------
    list[str] or None
        The user-specified element order if duplicates were found, else None.
    """

    unique_elements = list(dict.fromkeys(elements))

    if len(elements) != len(unique_elements):
        print("\nFound duplicated elements in POSCAR!")
        print("Unique elements: [" + " ".join(unique_elements) + "]")
        while True:
            sort_elements = input("Enter the desired element order (separate by space): ").split()
            if len(sort_elements) == 0:
                print("Warning! Empty input — using default unique element order.")
                return unique_elements.copy()
            if (len(sort_elements) == len(unique_elements) and
                    set(sort_elements) == set(unique_elements)):
                return sort_elements
            print("ERROR! The species do not match the unique elements. Try again.")
    else:
        return None


def mapping_elements(elements, atom_counts, positions_cartesian, positions_direct,
                     species, selective_dynamics, flags, sort_elements=None):
    """Re-order atoms so that each element block is contiguous and sorted canonically.

    Groups atomic positions by element symbol, resolves any duplicate element
    entries via check_elements(), and returns arrays sorted according to the
    specified (or user-supplied) element order. This is required because some
    POSCARs interleave atoms of the same species across multiple blocks.

    Parameters
    ----------
    elements            : list[str]            — element symbols from POSCAR
    atom_counts         : list[int]            — atoms per element block
    positions_cartesian : np.ndarray (N, 3)    — Cartesian coordinates in Å
    positions_direct    : np.ndarray (N, 3)    — fractional coordinates
    species             : list[str]            — per-atom element labels
    selective_dynamics  : bool                 — whether Selective Dynamics is used
    flags               : np.ndarray or None   — per-atom T/F flags
    sort_elements       : list[str] or None    — explicit element order (optional)

    Returns
    -------
    dict with keys:
        elements            : list[str]
        atom_counts         : list[int]
        positions_cartesian : np.ndarray (N, 3)
        positions_direct    : np.ndarray (N, 3)
        species             : list[str]
        flags               : np.ndarray or None
    """

    new_elements = elements.copy()
    new_atom_counts = atom_counts.copy()
    new_positions_cartesian = positions_cartesian.copy()
    new_positions_direct = positions_direct.copy()
    new_species = list(species).copy()
    new_flags = flags.copy() if (selective_dynamics and flags is not None) else None

    # Group positions and flags by element symbol
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
        if selective_dynamics and new_flags is not None:
            elements_flags.setdefault(element, []).extend(
                new_flags[position_index:position_index + count])
        position_index += count

    # Resolve canonical element order (prompts user if duplicates exist)
    if sort_elements is None:
        sort_elements = check_elements(elements)

    # Rebuild arrays in the resolved order
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
            if selective_dynamics and elements_flags is not None:
                sort_flags.extend(elements_flags[element])
            sort_atom_counts.append(len(elements_positions_direct[element]))

        new_positions_cartesian = np.array(sort_positions_cartesian, dtype=float)
        new_positions_direct = np.array(sort_positions_direct, dtype=float)
        new_species = list(sort_species)
        if selective_dynamics and sort_flags is not None:
            new_flags = np.array(sort_flags)
        new_atom_counts = sort_atom_counts
        new_elements = sort_elements

    return {"elements":           new_elements,
            "atom_counts":        new_atom_counts,
            "positions_cartesian": new_positions_cartesian,
            "positions_direct":   new_positions_direct,
            "species":            new_species,
            "flags":              new_flags if selective_dynamics else None}


def define_labels(elements, atom_counts):
    """Generate per-atom labels used as comments in the POSCAR position block.

    Labels take the form '<Symbol><index>' with the index zero-padded to the
    width of the largest atom count plus one (e.g. 'Mo01', 'S003').

    Parameters
    ----------
    elements    : list[str]  — element symbols in canonical order
    atom_counts : list[int]  — number of atoms per element

    Returns
    -------
    labels : list[str] — one label per atom in the same order as the position arrays
    """

    digits = len(str(max(atom_counts))) + 1
    labels = [f"{symbol}{str(counter).zfill(digits)}"
              for symbol, number in zip(elements, atom_counts)
              for counter in range(1, number + 1)]

    return labels


def write_POSCAR(filepath, lattice_matrix, elements, atom_counts,
                 positions_direct, selective_dynamics, flags, labels):
    """Write a VASP5-format POSCAR file with Direct coordinates.

    The scale factor is always written as 1.0 because the lattice vectors
    are already stored in absolute Å units. Atom labels are appended as
    inline comments after each position line for readability.

    Parameters
    ----------
    filepath           : str
    lattice_matrix     : np.ndarray (3, 3)  — lattice vectors in Å
    elements           : list[str]          — element symbols in canonical order
    atom_counts        : list[int]          — atoms per element
    positions_direct   : np.ndarray (N, 3)  — fractional coordinates
    selective_dynamics : bool
    flags              : np.ndarray or None  — per-atom T/F flags
    labels             : list[str]          — per-atom comment labels
    """

    with open(filepath, 'w') as o:
        o.write("Generated by vaspTwist.py code\n")
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
                        f"   {flag[0]:s}   {flag[1]:s}   {flag[2]:s}"
                        f"   {label:>6s}\n")
        else:
            for position, label in zip(positions_direct, labels):
                o.write(f"{position[0]:20.16f}{position[1]:20.16f}{position[2]:20.16f}"
                        f"   {label:>6s}\n")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def unwrap(positions_direct):
    """Reconstruct a contiguous cluster by unwrapping periodic boundary conditions.

    Shifts all atoms into the minimum-image frame relative to atom[0], so that
    atoms split across a cell boundary are treated as geometrically contiguous.
    Interatomic distances are preserved exactly.

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    reference : np.ndarray (3,)   — fractional coordinate of atom[0]
    unwrapped : np.ndarray (N, 3) — unwrapped fractional coordinates
    """

    reference = np.copy(positions_direct[0])
    delta = positions_direct - reference
    delta -= np.round(delta)

    return reference, reference + delta


def center_sheet(positions_direct):
    """Shift a 2D sheet so the vacuum direction is centered at 0.5 and the
    periodic directions start at origin.

    The vacuum axis is assumed to be z (index 2).

    Parameters
    ----------
    positions_direct : np.ndarray (N, 3) — fractional coordinates in [0, 1)

    Returns
    -------
    np.ndarray (N, 3) — shifted fractional coordinates in [0, 1)
    """

    reference, unwrapped = unwrap(positions_direct)
    center = np.mean(unwrapped, axis=0)
    vacuum = 2
    periodic = [i for i in range(3) if i != vacuum]
    new = np.copy(unwrapped)
    new[:, periodic] = unwrapped[:, periodic] - reference[periodic]
    new[:, vacuum]   = unwrapped[:, vacuum] - center[vacuum] + 0.5

    return new % 1.0


# ---------------------------------------------------------------------------
# Moiré supercell search (Numba-accelerated)
# ---------------------------------------------------------------------------

@njit
def rotation_matrix(degree):
    """Construct a rotation matrix for rotation about the z-axis.

    Uses the Rodrigues rotation formula:
        R = cos θ · I + sin θ · (u×) + (1 − cos θ) · u⊗u
    where u = [0, 0, 1].

    Parameters
    ----------
    degree : float

    Returns
    -------
    rotate : np.ndarray (3, 3) — rotation matrix
    """

    radian = np.radians(degree)
    sin = np.sin(radian)
    cos = np.cos(radian)
    u = np.array([0., 0., 1.])

    rotate = cos * np.eye(3) + sin * np.cross(np.eye(3), u) + (1 - cos) * np.outer(u, u)

    return rotate


@njit
def find_moire_vectors_chunk(bottom_lattice_matrix, top_lattice_matrix,
                              theta_chunk, combined_list, max_strain):
    """Search for near-coincident lattice vector pairs for a chunk of twist angles.

    Symmetric relative distance (CellMatch convention):
        ε = |v1 − v2| / (|v1| + |v2|)

    Parameters
    ----------
    bottom_lattice_matrix : np.ndarray (3, 3)
    top_lattice_matrix    : np.ndarray (3, 3)
    theta_chunk           : np.ndarray (K,)
    combined_list         : np.ndarray (M,)
    max_strain            : float

    Returns
    -------
    results : list of tuples (theta, n1, n2, rel_distance, v_layer1)
    """

    results = []
    for theta in theta_chunk:
        new_lattice_matrix = np.dot(top_lattice_matrix, rotation_matrix(theta).T)

        for n1 in combined_list:
            for n2 in combined_list:
                v_layer1 = n1 * bottom_lattice_matrix[0] + n2 * bottom_lattice_matrix[1]
                norm_v1 = np.linalg.norm(v_layer1)
                if norm_v1 == 0.0:
                    continue
                for m1 in combined_list:
                    for m2 in combined_list:
                        v_layer2 = m1 * new_lattice_matrix[0] + m2 * new_lattice_matrix[1]
                        norm_v2 = np.linalg.norm(v_layer2)
                        if norm_v2 == 0.0:
                            continue
                        # Symmetric relative distance (CellMatch convention)
                        rel_distance = np.linalg.norm(v_layer1 - v_layer2) / (norm_v1 + norm_v2)
                        if rel_distance <= max_strain:
                            results.append((theta, n1, n2, rel_distance, v_layer1))

    return results


def find_moire_vectors(bottom_lattice_matrix, top_lattice_matrix,
                        theta_min, theta_max, theta_step, n_min, n_max):
    """Search for commensurate moiré vectors over a range of twist angles.

    Parameters
    ----------
    bottom_lattice_matrix : np.ndarray (3, 3)
    top_lattice_matrix    : np.ndarray (3, 3)
    theta_min             : float
    theta_max             : float
    theta_step            : float
    n_min                 : int
    n_max                 : int

    Returns
    -------
    moire_vectors : list of tuples (theta, n1, n2, rel_distance, v_layer1)
    """

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
            [(bottom_lattice_matrix, top_lattice_matrix, chunk, combined_list, MAX_STRAIN)
             for chunk in theta_chunks]
        )

    moire_vectors = [item for sublist in chunk_results for item in sublist]

    return moire_vectors


# ---------------------------------------------------------------------------
# Supercell construction
# ---------------------------------------------------------------------------

def build_supercell(lattice_matrix, positions_cartesian, species,
                     selective_dynamics, flags, A1, A2):
    """Tile a primitive cell into a moiré supercell defined by vectors A1 and A2.

    Parameters
    ----------
    lattice_matrix      : np.ndarray (3, 3)
    positions_cartesian : np.ndarray (N, 3)
    species             : list[str]
    selective_dynamics  : bool
    flags               : np.ndarray or None
    A1, A2              : np.ndarray (3,)

    Returns
    -------
    dict with keys: lattice_matrix, positions_direct, species, flags
    """

    n_rep = int(np.ceil(max(np.linalg.norm(A1), np.linalg.norm(A2)) /
                        min(np.linalg.norm(lattice_matrix[0]),
                            np.linalg.norm(lattice_matrix[1])))) + 2

    supercell_positions_cartesian = []
    supercell_species = []
    supercell_flags = [] if selective_dynamics else None

    for n1 in range(-n_rep, n_rep + 1):
        for n2 in range(-n_rep, n_rep + 1):
            shift = n1 * lattice_matrix[0] + n2 * lattice_matrix[1]
            for atom_index in range(len(positions_cartesian)):
                supercell_positions_cartesian.append(positions_cartesian[atom_index] + shift)
                supercell_species.append(species[atom_index])
                if selective_dynamics:
                    supercell_flags.append(flags[atom_index])

    supercell_positions_cartesian = np.array(supercell_positions_cartesian)

    # New supercell lattice: A1, A2 in-plane, keep original c-axis
    new_lattice_matrix = np.array([A1, A2, lattice_matrix[2]])

    # Filter atoms inside the new supercell parallelogram
    new_positions_direct = np.dot(supercell_positions_cartesian,
                                  np.linalg.inv(new_lattice_matrix))
    inside_mask = ((new_positions_direct[:, 0] >= -1e-8) &
                   (new_positions_direct[:, 0] <   1.0 - 1e-8) &
                   (new_positions_direct[:, 1] >= -1e-8) &
                   (new_positions_direct[:, 1] <   1.0 - 1e-8))

    filtered_positions_direct = new_positions_direct[inside_mask] % 1.0
    filtered_species = [supercell_species[i]
                        for i in range(len(supercell_species)) if inside_mask[i]]
    filtered_flags = None
    if selective_dynamics:
        filtered_flags = np.array([supercell_flags[i]
                                   for i in range(len(supercell_flags)) if inside_mask[i]])

    return {"lattice_matrix":   new_lattice_matrix,
            "positions_direct": filtered_positions_direct,
            "species":          filtered_species,
            "flags":            filtered_flags if selective_dynamics else None}


def build_twisted_bilayer(layer1, layer2_rotated, selective_dynamics):
    """Stack two supercell layers into a twisted bilayer with a 3 Å interlayer gap.

    Parameters
    ----------
    layer1             : dict
    layer2_rotated     : dict
    selective_dynamics : bool

    Returns
    -------
    dict with keys: lattice_matrix, positions_direct, species, flags
    """

    layer1_cartesian = direct_to_cartesian(layer1["lattice_matrix"],
                                           layer1["positions_direct"])
    layer2_cartesian = direct_to_cartesian(layer2_rotated["lattice_matrix"],
                                           layer2_rotated["positions_direct"])

    z_max_layer1 = np.max(layer1_cartesian[:, 2])
    z_min_layer2 = np.min(layer2_cartesian[:, 2])
    interlayer_gap = 3.0
    layer2_cartesian[:, 2] += z_max_layer1 - z_min_layer2 + interlayer_gap

    combined_cartesian = np.vstack((layer1_cartesian, layer2_cartesian))
    combined_species = list(layer1["species"]) + list(layer2_rotated["species"])

    combined_flags = None
    if selective_dynamics and layer1["flags"] is not None and layer2_rotated["flags"] is not None:
        combined_flags = np.vstack((layer1["flags"], layer2_rotated["flags"]))

    combined_positions_direct = cartesian_to_direct(layer1["lattice_matrix"],
                                                    combined_cartesian)
    combined_positions_direct = center_sheet(combined_positions_direct)

    return {"lattice_matrix":   layer1["lattice_matrix"],
            "positions_direct": combined_positions_direct,
            "species":          combined_species,
            "flags":            combined_flags if selective_dynamics else None}


def collect_elements_and_counts(species, known_elements):
    """Collect unique elements present in species and their counts.

    Parameters
    ----------
    species        : list[str]
    known_elements : list[str]

    Returns
    -------
    elements_order : list[str]
    atom_counts    : list[int]
    """

    elements_order = list(dict.fromkeys(e for e in known_elements if e in species))
    atom_counts = [species.count(e) for e in elements_order]

    return elements_order, atom_counts


# ---------------------------------------------------------------------------
# Strain and geometry validation
# ---------------------------------------------------------------------------

def metric_tensor(e1, e2, e3):
    """Compute the 3×3 metric tensor G where G[i,j] = ei · ej.

    Parameters
    ----------
    e1, e2, e3 : np.ndarray (3,)

    Returns
    -------
    G : np.ndarray (3, 3)
    """

    vecs = [e1, e2, e3]
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            G[i, j] = np.dot(vecs[i], vecs[j])

    return G


def calculate_strain(A1, A2, a1, a2):
    """Compute the Lagrangian finite strain between a supercell and its primitive cell.

    Builds 3×3 metric tensors G = [ei · ej] for both vector sets, computes
    the Cholesky decomposition G = Rᵀ R, then evaluates the deformation
    gradient F = R_prim · R_super⁻¹ − I and the Lagrangian strain tensor
    ε = ½(F + Fᵀ + FᵀF). The scalar deformation measure is the RMS of the
    eigenvalues: deformation = √(Σλᵢ² / 3).

    The z-component of all vectors is set to 1.0 to avoid singularity
    (CellMatch convention for 2D systems).

    Parameters
    ----------
    A1, A2 : np.ndarray (3,) — supercell lattice vectors in Å
    a1, a2 : np.ndarray (3,) — primitive lattice vectors in Å

    Returns
    -------
    deformation : float — scalar Lagrangian strain measure
    """

    V1 = np.array([A1[0], A1[1], 0.0])
    V2 = np.array([A2[0], A2[1], 0.0])
    V3 = np.array([0.0,   0.0,   1.0])
    U1 = np.array([a1[0], a1[1], 0.0])
    U2 = np.array([a2[0], a2[1], 0.0])
    U3 = np.array([0.0,   0.0,   1.0])

    G1 = metric_tensor(V1, V2, V3)
    G2 = metric_tensor(U1, U2, U3)

    R1 = np.linalg.cholesky(G1).T
    R2 = np.linalg.cholesky(G2).T

    F = np.dot(R2, np.linalg.inv(R1)) - np.eye(3)
    lagrangian = 0.5 * (F + F.T + np.dot(F.T, F))

    eigenvalues = np.linalg.eigvalsh(lagrangian)
    deformation = np.sqrt(np.sum(eigenvalues ** 2) / 3.0)

    return float(deformation)


def calculate_area_ratio(A1, A2, a1, a2):
    """Compute the ratio of supercell area to primitive cell area.

    Parameters
    ----------
    A1, A2 : np.ndarray (3,)
    a1, a2 : np.ndarray (3,)

    Returns
    -------
    float
    """

    supercell_area = abs(A1[0] * A2[1] - A1[1] * A2[0])
    primitive_area = abs(a1[0] * a2[1] - a1[1] * a2[0])

    if primitive_area < 1e-12:
        return 0.0

    return supercell_area / primitive_area


# ---------------------------------------------------------------------------
# Candidate builder (one theta per worker process)
# ---------------------------------------------------------------------------

def build_candidates_for_theta(theta_key, vec_list,
                                bottom_lattice_matrix,
                                bottom_positions_cartesian,
                                bottom_species,
                                bottom_selective_dynamics,
                                bottom_flags,
                                top_lattice_matrix,
                                top_positions_cartesian,
                                top_species,
                                top_selective_dynamics,
                                top_flags,
                                sort_elements,
                                known_elements):
    """Build and filter bilayer supercell candidates for a single twist angle.

    Supports both homobilayer (bottom == top) and heterobilayer (different layers).

    Parameters
    ----------
    theta_key                   : float
    vec_list                    : list of (rel_distance, v_layer1)
    bottom_lattice_matrix       : np.ndarray (3, 3)
    bottom_positions_cartesian  : np.ndarray (N, 3)
    bottom_species              : list[str]
    bottom_selective_dynamics   : bool
    bottom_flags                : np.ndarray or None
    top_lattice_matrix          : np.ndarray (3, 3)
    top_positions_cartesian     : np.ndarray (M, 3)
    top_species                 : list[str]
    top_selective_dynamics      : bool
    top_flags                   : np.ndarray or None
    sort_elements               : list[str] or None
    known_elements              : list[str]

    Returns
    -------
    theta_candidates : list[dict]
    """

    a1 = bottom_lattice_matrix[0]
    a2 = bottom_lattice_matrix[1]

    rotated_top_lattice = np.dot(top_lattice_matrix, rotation_matrix(theta_key).T)

    selective_dynamics = bottom_selective_dynamics or top_selective_dynamics

    theta_candidates = []
    seen_configurations = set()

    for i in range(len(vec_list)):
        rel_dist_i, A1_vec = vec_list[i]
        for j in range(i + 1, len(vec_list)):
            rel_dist_j, A2_vec = vec_list[j]

            # Require non-degenerate supercell (A1 and A2 not parallel)
            cross_z = A1_vec[0] * A2_vec[1] - A1_vec[1] * A2_vec[0]
            if np.abs(cross_z) < 1e-6:
                continue

            # Reject supercells with too-sharp or too-blunt angles between A1 and A2
            # (30 to 150 degrees, matching MLM ang_lat concept but auto-ranged)
            norm_A1 = np.linalg.norm(A1_vec)
            norm_A2 = np.linalg.norm(A2_vec)
            if norm_A1 == 0.0 or norm_A2 == 0.0:
                continue
            cos_angle = np.dot(A1_vec, A2_vec) / (norm_A1 * norm_A2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_A1_A2 = np.degrees(np.arccos(cos_angle))
            if angle_A1_A2 < 30.0 or angle_A1_A2 > 150.0:
                continue

            # Validate supercell area is an integer multiple of bottom primitive cell area
            area_ratio = calculate_area_ratio(A1_vec, A2_vec, a1, a2)
            if abs(area_ratio - round(area_ratio)) > 0.1 or area_ratio < 0.5:
                continue

            config_key = (round(norm_A1, 3), round(norm_A2, 3))
            if config_key in seen_configurations:
                continue
            seen_configurations.add(config_key)

            lagrangian_strain = calculate_strain(A1_vec, A2_vec, a1, a2)

            layer1 = build_supercell(bottom_lattice_matrix,
                                      bottom_positions_cartesian,
                                      bottom_species,
                                      bottom_selective_dynamics,
                                      bottom_flags,
                                      A1_vec, A2_vec)

            layer2_rotated = build_supercell(rotated_top_lattice,
                                              top_positions_cartesian,
                                              top_species,
                                              top_selective_dynamics,
                                              top_flags,
                                              A1_vec, A2_vec)

            bilayer = build_twisted_bilayer(layer1, layer2_rotated,
                                             selective_dynamics)

            elements_order, atom_counts = collect_elements_and_counts(
                bilayer["species"], known_elements)

            bilayer_cartesian = direct_to_cartesian(bilayer["lattice_matrix"],
                                                    bilayer["positions_direct"])

            mapping = mapping_elements(elements_order, atom_counts,
                                        bilayer_cartesian,
                                        bilayer["positions_direct"],
                                        bilayer["species"],
                                        selective_dynamics,
                                        bilayer["flags"],
                                        sort_elements)

            total_atoms_bilayer = sum(mapping["atom_counts"])
            if total_atoms_bilayer > MAX_ATOMS:
                continue

            theta_candidates.append({"theta":                  theta_key,
                                      "A1_vec":                 A1_vec,
                                      "A2_vec":                 A2_vec,
                                      "strain":                 lagrangian_strain,
                                      "area_ratio":             int(round(area_ratio)),
                                      "n_bottom":               len(layer1["species"]),
                                      "mapping":                mapping,
                                      "bilayer_lattice_matrix": bilayer["lattice_matrix"],
                                      "total_atoms":            total_atoms_bilayer,
                                      "selective_dynamics":     selective_dynamics})

    return theta_candidates


# ---------------------------------------------------------------------------
# Candidate selection prompt
# ---------------------------------------------------------------------------

def prompt_selection(candidates):
    """Display a numbered table of candidates and prompt the user to select which to generate.

    Shows how many POSCAR files will be written per candidate (one per stacking point).
    The user can enter specific indices, 'all', or 'none' to skip generation.

    Parameters
    ----------
    candidates : list[dict]

    Returns
    -------
    chosen : list[int] — zero-based indices of selected candidates (empty list if none)
    """

    print(f"\nFound {len(candidates)} candidate(s) with <= {MAX_ATOMS} atoms:\n")
    print(f"  {'No.':>4}  {'Theta (deg)':>12}  {'Total atoms':>11}"
          f"  {'|A1| (Ang)':>10}  {'|A2| (Ang)':>10}  {'Area ratio':>10}"
          f"  {'Strain (%)':>10}  {'Stackings':>9}")
    print("  " + "-" * 97)
    for index, candidate in enumerate(candidates):
        theta       = candidate["theta"]
        total_atoms = candidate["total_atoms"]
        norm_A1     = np.linalg.norm(candidate["A1_vec"])
        norm_A2     = np.linalg.norm(candidate["A2_vec"])
        area_ratio  = candidate["area_ratio"]
        strain_pct  = candidate["strain"] * 100.0
        lattice_type   = detect_lattice_type(candidate["bilayer_lattice_matrix"])
        n_stackings    = len(get_stacking_shifts(lattice_type))
        print(f"  {index + 1:>4}  {theta:>12.4f}  {total_atoms:>11}"
              f"  {norm_A1:>10.4f}  {norm_A2:>10.4f}  {area_ratio:>10}"
              f"  {strain_pct:>9.4f}%  {n_stackings:>4} ({lattice_type})")

    total_poscars = sum(
        len(get_stacking_shifts(detect_lattice_type(c["bilayer_lattice_matrix"])))
        for c in candidates
    )
    print(f"\n  Selecting 'all' will write {total_poscars} POSCAR(s) total.")
    print( "  Enter 'none' to finish without generating any POSCAR.\n")

    while True:
        raw = input("Enter candidate numbers to generate (e.g. 1 3 5), 'all', or 'none': ").strip()
        if raw.lower() == 'none':
            return []
        if raw.lower() == 'all':
            return list(range(len(candidates)))
        try:
            chosen = [int(x) - 1 for x in raw.split()]
            if all(0 <= i < len(candidates) for i in chosen):
                return chosen
            print(f"ERROR! Numbers must be between 1 and {len(candidates)}.")
        except ValueError:
            print("ERROR! Enter integers separated by spaces, 'all', or 'none'.")


# ---------------------------------------------------------------------------
# Lattice type detection and stacking shift generation
# ---------------------------------------------------------------------------

def detect_lattice_type(lattice_matrix):
    """Classify the 2D Bravais lattice type from the in-plane lattice vectors.

    Compares the in-plane lattice lengths (a, b) and the angle γ between them
    to identify the crystal system according to the standard 2D classification:

        square      : γ = 90°  and  a = b
        rectangular : γ = 90°  and  a ≠ b
        hexagonal   : γ = 60° or 120°  and  a = b
        oblique     : all other cases

    Parameters
    ----------
    lattice_matrix : np.ndarray, shape (3, 3) — row vectors of the lattice in Å

    Returns
    -------
    str : one of 'square', 'rectangular', 'hexagonal', 'oblique'
    """

    length_a = np.linalg.norm(lattice_matrix[0])
    length_b = np.linalg.norm(lattice_matrix[1])
    gamma = np.degrees(np.arccos(np.clip(
        np.dot(lattice_matrix[0], lattice_matrix[1]) /
        (length_a * length_b), -1., 1.)))

    if np.abs(gamma - 90.) < 1e-5:
        return 'square' if np.abs(length_a - length_b) < 1e-8 else 'rectangular'
    elif ((np.abs(gamma - 60.) < 1e-5 or np.abs(gamma - 120.) < 1e-5)
          and np.abs(length_a - length_b) < 1e-8):
        return 'hexagonal'
    else:
        return 'oblique'


def get_stacking_shifts(lattice_type):
    """Return the canonical high-symmetry stacking shift points for a lattice type.

    Shifts are given in fractional coordinates of the supercell (A1, A2).
    The first point is always (0, 0) — the AA stacking reference.

    Parameters
    ----------
    lattice_type : str — output of detect_lattice_type

    Returns
    -------
    list of (float, float, str) — (shift_a, shift_b, label)
    """

    if lattice_type == "hexagonal":
        return [(0.0,       0.0,       "AA"),
                (1.0 / 3.0, 2.0 / 3.0, "AB"),
                (2.0 / 3.0, 1.0 / 3.0, "AB_prime")]

    elif lattice_type == "square":
        return [(0.0, 0.0, "AA"),
                (0.5, 0.0, "AB"),
                (0.5, 0.5, "AA_prime")]

    else:
        # rectangular or oblique
        return [(0.0, 0.0, "AA"),
                (0.5, 0.0, "AB_x"),
                (0.0, 0.5, "AB_y"),
                (0.5, 0.5, "AB_xy")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    if '-h' in argv or '--help' in argv or len(argv) not in (2, 3):
        usage()

    working_dir = os.getcwd()
    bottom = read_POSCAR(argv[1])
    top    = read_POSCAR(argv[2]) if len(argv) == 3 else bottom
    is_hetero = len(argv) == 3

    if is_hetero:
        print(f"\nHeterobilayer mode: {argv[1]}  +  {argv[2]}")
    else:
        print(f"\nHomobilayer mode: {argv[1]}")

    bilayer_elements = bottom["elements"] + top["elements"]
    sort_elements = check_elements(bilayer_elements)
    known_elements = sort_elements if sort_elements is not None else bilayer_elements
    selective_dynamics = bottom["selective_dynamics"] or top["selective_dynamics"]

    print(f"Searching for commensurate moiré vectors (0 to 180 deg, step = {THETA_STEP} deg)...")
    moire_vectors = find_moire_vectors(bottom["lattice_matrix"],
                                        top["lattice_matrix"],
                                        0.0, 180.0,
                                        THETA_STEP,
                                        N_MIN, N_MAX)

    if len(moire_vectors) == 0:
        print("No commensurate moiré vectors found with the given parameters.")
        exit(0)

    print(f"Found {len(moire_vectors)} raw result(s). Filtering candidates (<= {MAX_ATOMS} atoms)...\n")

    # Group coincident vectors by theta
    vectors_by_theta = {}
    for result in moire_vectors:
        theta    = result[0]
        rel_dist = result[3]
        vec      = np.array(result[4])
        vectors_by_theta.setdefault(round(theta, 4), []).append((rel_dist, vec))

    num_cores = mp.cpu_count()
    with mp.Pool(processes=num_cores) as pool:
        per_theta_results = pool.starmap(
            build_candidates_for_theta,
            [(theta_key, vec_list,
              bottom["lattice_matrix"],
              bottom["positions_cartesian"],
              bottom["species"],
              bottom["selective_dynamics"],
              bottom["flags"],
              top["lattice_matrix"],
              top["positions_cartesian"],
              top["species"],
              top["selective_dynamics"],
              top["flags"],
              sort_elements,
              known_elements)
             for theta_key, vec_list in vectors_by_theta.items()]
        )

    candidates = [c for theta_list in per_theta_results for c in theta_list]

    # Keep only the candidate with fewest atoms for each theta
    best_per_theta = {}
    for c in candidates:
        theta_key = c["theta"]
        if (theta_key not in best_per_theta or
                c["total_atoms"] < best_per_theta[theta_key]["total_atoms"]):
            best_per_theta[theta_key] = c
    candidates = list(best_per_theta.values())

    if len(candidates) == 0:
        print(f"No candidates found with <= {MAX_ATOMS} atoms. Try relaxing the parameters.")
        exit(0)

    candidates.sort(key=lambda c: (c["total_atoms"], c["theta"]))

    chosen_indices = prompt_selection(candidates)

    if len(chosen_indices) == 0:
        print("\nNo candidates selected. Finished without writing any POSCAR.\n")
        exit(0)

    written_count = 0
    for index in chosen_indices:
        candidate      = candidates[index]
        theta          = candidate["theta"]
        mapping        = candidate["mapping"]
        lattice_matrix = candidate["bilayer_lattice_matrix"]
        total_atoms    = candidate["total_atoms"]
        area_ratio     = candidate["area_ratio"]
        strain_pct     = candidate["strain"] * 100.0
        n_bottom       = candidate["n_bottom"]

        labels = define_labels(mapping["elements"], mapping["atom_counts"])

        # Detect lattice type from supercell and get stacking shifts
        lattice_type    = detect_lattice_type(lattice_matrix)
        stacking_shifts = get_stacking_shifts(lattice_type)

        base_dir = os.path.join(working_dir, f"twist_{theta:.4f}deg_{total_atoms}atoms")
        os.makedirs(base_dir, exist_ok=True)

        print(f"\n  theta = {theta:.4f} deg | {total_atoms} atoms"
              f" | area ratio = {area_ratio} | strain = {strain_pct:.4f}%"
              f" | lattice type = {lattice_type}")

        for shift_a, shift_b, stack_label in stacking_shifts:

            # Apply shift to top-layer atoms only
            # n_bottom is the exact count from build_candidates_for_theta
            shifted_positions = mapping["positions_direct"].copy()
            shifted_positions[n_bottom:, 0] = (shifted_positions[n_bottom:, 0] + shift_a) % 1.0
            shifted_positions[n_bottom:, 1] = (shifted_positions[n_bottom:, 1] + shift_b) % 1.0

            output_dir  = os.path.join(base_dir, stack_label)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "POSCAR")

            write_POSCAR(output_path,
                         lattice_matrix,
                         mapping["elements"],
                         mapping["atom_counts"],
                         shifted_positions,
                         selective_dynamics,
                         mapping["flags"],
                         labels)

            print(f"    Written: {output_dir}/POSCAR  (stacking = {stack_label},"
                  f" shift = ({shift_a:.4f}, {shift_b:.4f}))")
            written_count += 1

    print(f"\nFinished! Written {written_count} POSCAR(s).\n")


if __name__ == "__main__":
    main()
