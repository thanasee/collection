#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np

def usage():
    
    text = """
Usage: StrainEnergy2D.py <mode> [structure file]

This script support VASP5 Structure file format (i.e. POSCAR) 
for applied strain to position file.

Mode:
- pre  <structure_file> : prepare structure files for elastic tensor calculations
- post                  : get energy and calculate elastic tensor

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

ANGSTROM = 1e-10
ALL_STRAIN = ['C11', 'C22', 'C11_C22_2C12', 'C66']
STRAIN_RANGE = np.linspace(-0.05, 0.05, 11)
ZERO_STRAIN_IDX = 5

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
                 selective_dynamics,
                 positions_direct,
                 flags,
                 labels):
    
    with open(filepath, 'w') as o:
        o.write("Generated by StrainEnergy2D.py code\n")
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

def get_2d_lattice_type(lattice_matrix):
    
    length_a = np.linalg.norm(lattice_matrix[0])
    length_b = np.linalg.norm(lattice_matrix[1])
    gamma = np.degrees(np.arccos(np.clip(np.dot(lattice_matrix[0], lattice_matrix[1]) / (np.linalg.norm(lattice_matrix[0]) * np.linalg.norm(lattice_matrix[1])), -1., 1.)))
    
    
    if np.abs(gamma - 90.) < 1e-5:
        return 'square' if np.abs(length_a - length_b) < 1e-8 else 'rectangular'
    elif (np.abs(gamma - 60.) < 1e-5 or np.abs(gamma - 120.) < 1e-5) and np.abs(length_a - length_b) < 1e-8:
        return 'hexagonal'
    else:
        return 'oblique'

def get_strain_types(crystal_system):
 
    strain_types = ALL_STRAIN.copy()
    if crystal_system == 'oblique':
        strain_types.extend(['C11_C66_2C16', 'C22_C66_2C26'])
        
    return strain_types

def build_strain_matrix(strain_type,
                        strain):
 
    strain_matrix = {'C11': np.array([[strain, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
                     'C22':  np.array([[0, 0, 0], [0, strain, 0], [0, 0, 0]], dtype=float),
                     'C11_C22_2C12': np.array([[strain, 0, 0], [0, strain, 0], [0, 0, 0]], dtype=float),
                     'C66': np.array([[0, strain / 2, 0], [strain / 2, 0, 0], [0, 0, 0]], dtype=float),
                     'C11_C66_2C16': np.array([[strain, strain / 2, 0], [strain / 2, 0, 0], [0, 0, 0]], dtype=float),
                     'C22_C66_2C26': np.array([[0, strain / 2, 0], [strain / 2, strain, 0], [0, 0, 0]], dtype=float)}
 
    return strain_matrix[strain_type]

def applying_strain(lattice_matrix,
                    strain_matrix):
    
    return np.dot(lattice_matrix, np.eye(3) + strain_matrix)

def read_OUTCAR(filepath):
    
    if not os.path.exists(filepath):
        print(f"ERROR! File not found: {filepath}")
        return None, False
    
    with open(filepath, 'r') as f:
        outcar_lines = f.readlines()
    
    converged = any('reached required accuracy' in line for line in outcar_lines)
    
    if not converged:
        print(f"WARNING! {filepath} has not reached required accuracy.")
    
    energy = None
    for line in reversed(outcar_lines):
        if 'energy  without entropy=' in line:
            energy = float(line.split()[-1])
            break
    
    if energy is None:
        print(f"WARNING! Could not extract energy from {filepath}.")
 
    return energy, converged

def fitting_strain_energy(strain_type,
                         strain_range,
                         strain_energy,
                         area):
    
    from scipy.constants import e
    from scipy.optimize import curve_fit
    
    eVpA2_to_Npm = e / ANGSTROM ** 2
    
    def quadratic(x, a):
        return a * x ** 2
    
    energy_per_area = (strain_energy - strain_energy[ZERO_STRAIN_IDX]) / area
    
    output_file = f"StrainVsEnergy_{strain_type}.dat"
    with open(output_file, 'w') as o:
        o.write(f"# Strain vs Energy per area — {strain_type}\n")
        o.write("# Strain    U(eV/Å²)\n")
        for s, u in zip(strain_range, energy_per_area):
            o.write(f"  {s:>+6.2f}   {u:>14.6f}\n")

    coef, _ = curve_fit(quadratic, strain_range, energy_per_area)
    constant = 2 * coef[0] * eVpA2_to_Npm

    return constant

def collect_fitting_coef(strain_types,
                         strain_range,
                         area):
 
    constants = {}
 
    for strain_type in strain_types:
 
        strain_energy = []
        all_converged = True
 
        for strain in strain_range:
            strain_path = os.path.join(strain_type, f"strain{strain:+.2f}")
            outcar = os.path.join(strain_path, 'OUTCAR')
            energy, converged = read_OUTCAR(outcar)
            strain_energy.append(energy)
            if not converged:
                all_converged = False
 
        if not all_converged or None in strain_energy:
            print(f"WARNING! Skipping '{strain_type}' — incomplete or unconverged calculations.")
            constants[strain_type] = None
            continue
 
        strain_energy = np.array(strain_energy, dtype=float)
        constants[strain_type] = fitting_strain_energy(strain_type, strain_range, strain_energy, area)
 
    return constants

def obtain_elastic_tensor(constants,
                          crystal_system):
 
    missing = [k for k, v in constants.items() if v is None]
    if missing:
        for key in missing:
            print(f"ERROR! {key} is not calculated.")
        return None
 
    C11 = constants['C11']
    C22 = constants['C22']
    C66 = constants['C66']
    C12 = (constants['C11_C22_2C12'] - C11 - C22) / 2
    C16 = 0.
    C26 = 0.

    if crystal_system == 'oblique':
        C16 = (constants['C11_C66_2C16'] - C11 - C66) / 2
        C26 = (constants['C22_C66_2C26'] - C22 - C66) / 2
    
    elastic_tensor = np.array([[C11, C12, C16],
                               [C12, C22, C26],
                               [C16, C26, C66]])

    return elastic_tensor

def check_stability(elastic_tensor,
                    lattice_matrix,
                    area_vector,
                    area):

    vector_n  = area_vector / area
    factor_2d = np.abs(np.dot(lattice_matrix[2], vector_n)) / 10
    
    return np.all(np.linalg.eigvalsh(elastic_tensor) > 1e-5 * factor_2d)

def compute_mechanical_properties(elastic_tensor):
    
 
    degrees = np.arange(0, 360.0, 0.1)
    radians = np.deg2rad(degrees)
    sin = np.sin(radians)
    cos = np.cos(radians)
 
    compliance_tensor = np.linalg.inv(elastic_tensor)
    S11 = compliance_tensor[0, 0]
    S22 = compliance_tensor[1, 1]
    S12 = compliance_tensor[0, 1]
    S66 = compliance_tensor[2, 2]
    S16 = compliance_tensor[0, 2]
    S26 = compliance_tensor[1, 2]
 
    A = (S11 * cos**4 + S22 * sin**4  + (2 * S12 + S66) * cos**2 * sin**2 + 2 * S16 * cos**3 * sin + 2 * S26 * cos * sin**3)
    B = ((S11 + S22 - S66) * cos**2 * sin**2 + S12 * (cos**4 + sin**4) + (S26 - S16) * (cos**3 * sin - cos * sin**3))
    C = (4 * (S11 + S22 - 2 * S12) * cos**2 * sin**2 + S66 * (cos**2 - sin**2)**2 + 4 * (S16 - S26) * (cos**3 * sin - cos * sin**3))
 
    young_modulus  = 1 / A
    poisson_ratio  = -B / A
    shear_modulus  = 1 / C
 
    return {"degrees": degrees,
            "young_modulus":  young_modulus,
            "poisson_ratio":  poisson_ratio,
            "shear_modulus":  shear_modulus}

def write_mechanical_properties(properties):
 
    degrees = properties["degrees"]
    young_modulus  = properties["young_modulus"]
    poisson_ratio  = properties["poisson_ratio"]
    shear_modulus  = properties["shear_modulus"]

    with open('Young.dat', 'w') as o:
        o.write("# Young's Modulus\n")
        o.write("#  Degree(°)  Y(N/m)\n")
        for dg, y in zip(degrees, young_modulus):
            o.write(f" {dg:>6.2f}     {y:>12.8f}\n")

    with open('Poisson.dat', 'w') as o:
        o.write("# Poisson's Ratio\n")
        if (poisson_ratio < 0.).any():
            o.write("#  Degree(°) v             |v|\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.2f}   {nu:>12.8f}   {abs(nu):>12.8f}\n")
        else:
            o.write("#  Degree(°) v\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.2f}   {nu:>12.8f}\n")

    with open('Shear.dat', 'w') as o:
        o.write("# Shear Modulus\n")
        o.write("#  Degree(°)  G(N/m)\n")
        for dg, g in zip(degrees, shear_modulus):
            o.write(f" {dg:>6.2f}     {g:>12.8f}\n")

def write_elastic_tensor(elastic_tensor):

    C = elastic_tensor
    header = ("# Elastic tensor (N/m)\n"
              "#     C11         C12         C16\n"
              "#     C12         C22         C26\n"
              "#     C16         C26         C66\n\n")
    rows = (f"   {C[0,0]:>11.4f} {C[0,1]:>11.4f} {C[0,2]:>11.4f}\n"
            f"   {C[1,0]:>11.4f} {C[1,1]:>11.4f} {C[1,2]:>11.4f}\n"
            f"   {C[2,0]:>11.4f} {C[2,1]:>11.4f} {C[2,2]:>11.4f}\n")

    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)

    print("\n" + header + rows)

def mode_pre(filepath):

    poscar = read_POSCAR(filepath)
    crystal_system = get_2d_lattice_type(poscar["lattice_matrix"])
    strain_types = get_strain_types(crystal_system)

    print(f"\nDetected crystal system: {crystal_system}")
    print(f"Strain types to compute: {strain_types}\n")

    mapping = mapping_elements(poscar["elements"],
                               poscar["atom_counts"],
                               poscar["positions_cartesian"],
                               poscar["positions_direct"],
                               poscar["species"],
                               poscar["selective_dynamics"],
                               poscar["flags"])
    labels = define_labels(mapping["elements"],
                           mapping["atom_counts"])

    # Write unstrained reference structure
    unstrain_path = 'unstrain'
    os.makedirs(unstrain_path, exist_ok=True)
    write_POSCAR(os.path.join(unstrain_path, 'POSCAR'),
                 poscar["lattice_matrix"],
                 mapping["elements"],
                 mapping["atom_counts"],
                 poscar["selective_dynamics"],
                 mapping["positions_direct"],
                 mapping["flags"],
                 labels)
 
    # Write strained structures
    for strain_type in strain_types:
        for strain in STRAIN_RANGE:
            strain_path = os.path.join(strain_type, f"strain{strain:+.2f}")
            os.makedirs(strain_path, exist_ok=True)
            strain_matrix = build_strain_matrix(strain_type, strain)
            new_lattice_matrix = applying_strain(poscar["lattice_matrix"],
                                                 strain_matrix)
            write_POSCAR(os.path.join(strain_path, 'POSCAR'),
                         new_lattice_matrix,
                         mapping["elements"],
                         mapping["atom_counts"],
                         poscar["selective_dynamics"],
                         mapping["positions_direct"],
                         mapping["flags"],
                         labels)
 
    print(f"Done. Strained POSCARs written for {len(strain_types)} strain types × 11 strain values.\n")

def mode_post():
 
    unstrain_poscar = os.path.join('unstrain', 'POSCAR')
    poscar = read_POSCAR(unstrain_poscar)
 
    crystal_system = get_2d_lattice_type(poscar["lattice_matrix"])
    strain_types = get_strain_types(crystal_system)
 
    print(f"\nDetected crystal system: {crystal_system}")
 
    area_vector = np.cross(poscar["lattice_matrix"][0], poscar["lattice_matrix"][1])
    area = np.linalg.norm(area_vector)
 
    # Collect fitted elastic constants
    constants = collect_fitting_coef(strain_types, STRAIN_RANGE, area)
 
    # Assemble full elastic tensor
    elastic_tensor = obtain_elastic_tensor(constants,
                                           crystal_system)
    if elastic_tensor is None:
        print("ERROR! Elastic tensor could not be assembled. Exiting.")
        exit(1)
 
    write_elastic_tensor(elastic_tensor)
 
    # Mechanical stability
    stable = check_stability(elastic_tensor,
                             poscar["lattice_matrix"],
                             area_vector,
                             area)
    if stable:
        print("This material is mechanically stable.\n")
    else:
        print("This material is mechanically unstable!!\n")
        exit(0)
 
    # Angular mechanical properties
    properties = compute_mechanical_properties(elastic_tensor)
    write_mechanical_properties(properties)
    print("Mechanical properties written to Young.dat, Poisson.dat, Shear.dat.\n")

def main():

    if '-h' in argv or '--help' in argv or len(argv) < 2:
        usage()
    
    mode = argv[1]
    
    if mode == 'pre':
        if len(argv) < 3:
            print("ERROR! 'pre' mode requires a structure file as argument.")
            usage()
        mode_pre(argv[2])
    elif mode == 'post':
        mode_post()
    else:
        print(f"ERROR! Unknown mode: '{mode}'. Use 'pre' or 'post'.")
        usage()

if __name__ == "__main__":
    main()
