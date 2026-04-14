#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
from ase.io import read
from ase.spacegroup.symmetrize import check_symmetry
from scipy.constants import Avogadro, h, k

def usage():
    
    text = """
Usage: vaspMechanics.py <POSCAR input> <OUTCAR input>

This script calculates Young's modulus and Poisson's Ratio as functions of crystal orientation.
Output files can plot by Origin Program or plotMechanics.py scritpt. (For 2D)
Check mechanical stability too.
This script read POSCAR and OUTCAR files by default.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

def read_structure(poscar_file):
    
    if not os.path.exists(poscar_file):
        print(f"ERROR!\nFile: {poscar_file} does not exist.")
        exit(0)
        
    return read(poscar_file)

def read_elastic_tensor(outcar_file):
    
    if not os.path.exists(outcar_file):
        print(f"ERROR!\nFile: {outcar_file} does not exist.")
        exit(0)
    
    with open(outcar_file, 'r') as f:
        outcar_lines = f.readlines()
    
    moduli_index = None
    for i, line in enumerate(outcar_lines):
        if 'TOTAL ELASTIC MODULI (kBar)' in line:
            moduli_index = i
            break
    
    if moduli_index is None:
        print("The 'TOTAL ELASTIC MODULI (kBar)' section was not found in the OUTCAR file.")
        exit(0)
    
    moduli_lines = outcar_lines[moduli_index + 3:moduli_index + 9]
    elastic_vasp = np.array([[float(x) for x in line.split()[1:]]
                             for line in moduli_lines]) / 10 # Convert unit kBar to GPa
    
    # Reorder from VASP convention (xx,yy,zz,xy,yz,xz) to Voigt notation (11,22,33,44,55,66)
    elastic_coef = elastic_vasp[:, [0, 1, 2, 4, 5, 3]][[0, 1, 2, 4, 5, 3], :]
    
    return elastic_coef

def get_2d_lattice_type(structure):
    
    length_a = structure.cell.cellpar()[0]
    length_b = structure.cell.cellpar()[1]
    gamma    = structure.cell.cellpar()[5]
    
    if np.abs(gamma - 90.) < 1e-5:
        return 'square' if np.abs(length_a - length_b) < 1e-8 else 'rectangular'
    elif (np.abs(gamma - 60.) < 1e-5 or np.abs(gamma - 120.) < 1e-5) and np.abs(length_a - length_b) < 1e-8:
        return 'hexagonal'
    else:
        return 'oblique'

def compute_elastic_2d(structure, elastic_coef):
    
    vector_a = structure.cell[0]
    vector_b = structure.cell[1]
    vector_c = structure.cell[2]
    
    vector_n  = np.cross(vector_a, vector_b) / np.linalg.norm(np.cross(vector_a, vector_b))
    factor_2d = np.abs(np.dot(vector_c, vector_n)) / 10  # Angstrom to nm
    
    elastic_2d = elastic_coef[np.ix_([0, 1, 5], [0, 1, 5])] * factor_2d  # Convert unit GPa*Angstrom to N/m
    
    lattice_type = get_2d_lattice_type(structure)
    print(f"This material is {lattice_type.capitalize()}.")
    
    oblique = (lattice_type == 'oblique')
    
    C11 = elastic_2d[0, 0]
    C22 = elastic_2d[1, 1]
    C12 = elastic_2d[0, 1]
    C66 = elastic_2d[2, 2]
    C16 = elastic_2d[0, 2] if oblique else 0.
    C26 = elastic_2d[1, 2] if oblique else 0.
    
    return elastic_2d, C11, C22, C12, C66, C16, C26, factor_2d

def write_elastic_2d(C11,
                     C22,
                     C12,
                     C66,
                     C16,
                     C26):
    
    header = (
        "# Elastic tensor(N/m)\n"
        "#     C11         C12         C16\n"
        "#     C12         C22         C26\n"
        "#     C16         C26         C66\n\n"
    )
    rows = (
        f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}\n"
        f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}\n"
        f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n"
    )
    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)
    
    print("\n# Elastic tensor(N/m)")
    print("#     C11         C12         C16")
    print("#     C12         C22         C26")
    print("#     C16         C26         C66")
    print(f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}")
    print(f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}")
    print(f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n")


def check_stability_2d(elastic_2d,
                       factor_2d):
    if np.all(np.linalg.eigvalsh(elastic_2d) > 1e-5 * factor_2d):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(0)

def compute_directional_properties_2d(elastic_2d):
    degrees = np.arange(0, 360.0, 0.1)
    radians = np.deg2rad(degrees)
    sin = np.sin(radians)
    cos = np.cos(radians)
    
    compliance_2d = np.linalg.inv(elastic_2d)
    
    S11 = compliance_2d[0, 0]
    S22 = compliance_2d[1, 1]
    S12 = compliance_2d[0, 1]
    S66 = compliance_2d[2, 2]
    S16 = compliance_2d[0, 2]
    S26 = compliance_2d[1, 2]
    
    # Calculate Young's modulus, Poisson's ratio, and shear modulus from Sij
    A = (S11 * cos**4 + S22 * sin**4
         + (2 * S12 + S66) * cos**2 * sin**2
         + 2 * S16 * cos**3 * sin
         + 2 * S26 * cos * sin**3)
    young_modulus = 1 / A
    
    B = ((S11 + S22 - S66) * cos**2 * sin**2
         + S12 * (cos**4 + sin**4)
         + (S26 - S16) * (cos**3 * sin - cos * sin**3))
    poisson_ratio = -B / A
    
    C = (4 * (S11 + S22 - 2 * S12) * cos**2 * sin**2
         + S66 * (cos**2 - sin**2)**2
         + 4 * (S16 - S26) * (cos**3 * sin - cos * sin**3))
    shear_modulus = 1 / C
    
    return degrees, young_modulus, poisson_ratio, shear_modulus

def write_directional_properties_2d(degrees,
                                    young_modulus,
                                    poisson_ratio,
                                    shear_modulus):
    
    with open('Young.dat', 'w') as o:
        o.write("# Young's Modulus\n")
        o.write("#  Degree(\u00B0)  Y(N/m)\n")
        for dg, y in zip(degrees, young_modulus):
            o.write(f" {dg:>6.1f}      {y:>12.8f}\n")
    
    with open('Poisson.dat', 'w') as o:
        o.write("# Poisson's Ratio\n")
        if (poisson_ratio < 0.).any():
            o.write("#  Degree(\u00B0) v             |v|\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.1f}    {nu:>12.8f}   {abs(nu):>12.8f}\n")
        else:
            o.write("#  Degree(\u00B0) v\n")
            for dg, nu in zip(degrees, poisson_ratio):
                o.write(f" {dg:>6.1f}   {nu:>12.8f}\n")
    
    with open('Shear.dat', 'w') as o:
        o.write("# Shear modulus\n")
        o.write("#  Degree(\u00B0)  G(N/m)\n")
        for dg, g in zip(degrees, shear_modulus):
            o.write(f" {dg:>6.1f}      {g:>12.8f}\n")

def run_2d(structure, elastic_coef):
    
    elastic_2d, C11, C22, C12, C66, C16, C26, factor_2d = compute_elastic_2d(structure, elastic_coef)
    write_elastic_2d(C11, C22, C12, C66, C16, C26)
    check_stability_2d(elastic_2d, factor_2d)
    degrees, young_modulus, poisson_ratio, shear_modulus = compute_directional_properties_2d(elastic_2d)
    write_directional_properties_2d(degrees, young_modulus, poisson_ratio, shear_modulus)

def get_crystal_system(structure):
    spacegroup = check_symmetry(structure).number
    
    if 195 <= spacegroup <= 230:
        return 'Cubic'
    elif 168 <= spacegroup <= 194:
        return 'Hexagonal'
    elif 149 <= spacegroup <= 167:
        return 'Trigonal I'
    elif 143 <= spacegroup <= 148:
        return 'Trigonal II'
    elif 89 <= spacegroup <= 142:
        return 'Tetragonal I'
    elif 75 <= spacegroup <= 88:
        return 'Tetragonal II'
    elif 16 <= spacegroup <= 74:
        return 'Orthorhombic'
    elif 3 <= spacegroup <= 15:
        return 'Monoclinic'
    else:
        return 'Triclinic'

def write_elastic_3d(elastic_3d):
    C = elastic_3d  # shorthand for formatting
    
    header = (
        "# Elastic tensor(GPa)\n"
        "#     C11         C12         C13         C14         C15         C16\n"
        "#     C12         C22         C23         C24         C25         C26\n"
        "#     C13         C23         C33         C34         C35         C36\n"
        "#     C14         C24         C34         C44         C45         C46\n"
        "#     C15         C25         C35         C45         C55         C56\n"
        "#     C16         C26         C36         C46         C56         C66\n\n"
    )
    rows = ""
    for i in range(6):
        rows += "   " + " ".join(f"{C[i, j]:>11.4f}" for j in range(6)) + "\n"
    
    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)
    
    print("\n# Elastic tensor(GPa)")
    print("#     C11         C12         C13         C14         C15         C16")
    print("#     C12         C22         C23         C24         C25         C26")
    print("#     C13         C23         C33         C34         C35         C36")
    print("#     C14         C24         C34         C44         C45         C46")
    print("#     C15         C25         C35         C45         C55         C56")
    print("#     C16         C26         C36         C46         C56         C66")
    for i in range(6):
        print("   " + " ".join(f"{C[i, j]:>11.4f}" for j in range(6)))
    print()

def check_stability_3d(elastic_3d):
    if np.all(np.linalg.eigvalsh(elastic_3d) > 1e-5):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(0)

def compute_mechanical_properties_3d(elastic_3d, structure):
    C11 = elastic_3d[0, 0]; C22 = elastic_3d[1, 1]; C33 = elastic_3d[2, 2]
    C12 = elastic_3d[0, 1]; C13 = elastic_3d[0, 2]; C23 = elastic_3d[1, 2]
    C44 = elastic_3d[3, 3]; C55 = elastic_3d[4, 4]; C66 = elastic_3d[5, 5]
    
    compliance_3d = np.linalg.inv(elastic_3d)
    
    S11 = compliance_3d[0, 0]; S22 = compliance_3d[1, 1]; S33 = compliance_3d[2, 2]
    S12 = compliance_3d[0, 1]; S13 = compliance_3d[0, 2]; S23 = compliance_3d[1, 2]
    S44 = compliance_3d[3, 3]; S55 = compliance_3d[4, 4]; S66 = compliance_3d[5, 5]
    
    total_mass  = structure.get_masses().sum()  # amu
    volume      = structure.get_volume()         # Angstrom^3
    density     = total_mass / volume            # amu / Angstrom^3
    total_atoms = len(structure)
    
    # VRH averages
    bulk_voigt  = (C11 + C22 + C33 + 2 * (C12 + C23 + C13)) / 9
    shear_voigt = (C11 + C22 + C33 - C12 - C23 - C13 + 3 * (C44 + C55 + C66)) / 15
    bulk_reuss  = 1 / (S11 + S22 + S33 + 2 * (S12 + S23 + S13))
    shear_reuss = 15 / (4 * (S11 + S22 + S33 - S12 - S23 - S13) + 3 * (S44 + S55 + S66))
    bulk_modulus  = (bulk_voigt + bulk_reuss) / 2
    shear_modulus = (shear_voigt + shear_reuss) / 2
    
    young_modulus  = (9 * bulk_modulus * shear_modulus) / (3 * bulk_modulus + shear_modulus)
    pugh_ratio     = bulk_modulus / shear_modulus
    poisson_ratio  = (3 * bulk_modulus - 2 * shear_modulus) / (2 * (3 * bulk_modulus + shear_modulus))
    pwave_modulus  = bulk_modulus + 4 * shear_modulus / 3
    lame_parameter = bulk_modulus - 2 * shear_modulus / 3
    
    # Anisotropy indices
    universal_anisotropy = 5 * (shear_voigt / shear_reuss) + (bulk_voigt / bulk_reuss) - 6
    bulk_anisotropy      = (bulk_voigt - bulk_reuss) / (bulk_voigt + bulk_reuss)
    shear_anisotropy     = (shear_voigt - shear_reuss) / (shear_voigt + shear_reuss)
    anisotropy_1 = 4 * C44 / (C11 + C33 - 2 * C13)
    anisotropy_2 = 4 * C55 / (C22 + C33 - 2 * C23)
    anisotropy_3 = 4 * C66 / (C11 + C22 - 2 * C12)
    
    # Unit conversions
    unit_modulus = 1e9           # GPa to Pa
    unit_mass    = 1e-3 / Avogadro  # amu to kg
    unit_volume  = (1e-10)**3    # Angstrom^3 to m^3
    unit_density = unit_mass / unit_volume  # kg / m^3
    
    # Sound velocities
    v_t = np.sqrt((shear_modulus * unit_modulus) / (density * unit_density))                               # m/s
    v_l = np.sqrt(((3 * bulk_modulus + 4 * shear_modulus) * unit_modulus) / (3 * density * unit_density))  # m/s
    v_m = np.cbrt(((2 / v_t**3) + (1 / v_l**3)) / 3) ** -1                                                # m/s
    
    debye_temperature = (h / k) * np.cbrt((3 * total_atoms) / (4 * np.pi * volume * unit_volume)) * v_m   # K
    
    return {
       'bulk_voigt': bulk_voigt, 'bulk_reuss': bulk_reuss,
       'shear_voigt': shear_voigt, 'shear_reuss': shear_reuss,
       'bulk_modulus': bulk_modulus, 'shear_modulus': shear_modulus,
       'young_modulus': young_modulus, 'poisson_ratio': poisson_ratio,
       'pwave_modulus': pwave_modulus, 'lame_parameter': lame_parameter,
       'pugh_ratio': pugh_ratio, 'v_m': v_m, 'debye_temperature': debye_temperature,
       'universal_anisotropy': universal_anisotropy,
       'bulk_anisotropy': bulk_anisotropy, 'shear_anisotropy': shear_anisotropy,
       'anisotropy_1': anisotropy_1, 'anisotropy_2': anisotropy_2, 'anisotropy_3': anisotropy_3}

def print_and_write_mechanical_properties_3d(props):
    bv = props['bulk_voigt']; br = props['bulk_reuss']
    gv = props['shear_voigt']; gr = props['shear_reuss']
    B = props['bulk_modulus']; G = props['shear_modulus']
    E = props['young_modulus']; v = props['poisson_ratio']
    M = props['pwave_modulus']; L = props['lame_parameter']
    P = props['pugh_ratio']; vm = props['v_m']
    tD = props['debye_temperature']
    A_B = props['bulk_anisotropy']; A_G = props['shear_anisotropy']
    A_U = props['universal_anisotropy']
    A1 = props['anisotropy_1']; A2 = props['anisotropy_2']; A3 = props['anisotropy_3']
    
    mech_str = (
        f"\n# Mechanical properties\n"
        f"Voigt's Bulk modulus          (B_V): {bv:>10.4f} GPa\n"
        f"Reuss' Bulk modulus           (B_R): {br:>10.4f} GPa\n"
        f"Voigt's Shear Modulus         (G_V): {gv:>10.4f} GPa\n"
        f"Reuss' Shear Modulus          (G_R): {gr:>10.4f} GPa\n"
        f"Bulk Modulus                  (B)  : {B:>10.4f} GPa\n"
        f"Shear Modulus                 (G)  : {G:>10.4f} GPa\n"
        f"Young's Modulus               (E)  : {E:>10.4f} GPa\n"
        f"Poisson's Ratio               (v)  : {v:>10.4f}\n"
        f"P-wave Modulus                (M)  : {M:>10.4f} GPa\n"
        f"Lam\u00e9's first parameter        (L)  : {L:>10.4f} GPa\n"
        f"Pugh's Ratio                  (P)  : {P:>10.4f}\n"
        f"Sound velocity                (v_m): {vm:>10.4f} m/s\n"
        f"Debye temperature             (\u03B8_D): {tD:>10.4f} K\n"
    )
    
    aniso_str = (
        f"\n# Anisotropic analysis\n"
        f"Bulk Anisotropy               (A_B): {A_B:>10.4f}\n"
        f"Shear Anisotropy              (A_G): {A_G:>10.4f}\n"
        f"Universal Anisotropy          (A_U): {A_U:>10.4f}\n"
        f"(100) Planar Shear Anisotropy (A_1): {A1:>10.4f}\n"
        f"(010) Planar Shear Anisotropy (A_2): {A2:>10.4f}\n"
        f"(001) Planar Shear Anisotropy (A_3): {A3:>10.4f}\n"
    )
    
    print(mech_str + aniso_str)
    
    with open('Mechanics.dat', 'w') as o:
        o.write("# Mechanical properties\n")
        o.write(f"Bulk Modulus                  (B)  : {B:>10.4f} GPa\n")
        o.write(f"Voigt's Bulk modulus          (B_V): {bv:>10.4f} GPa\n")
        o.write(f"Reuss' Bulk modulus           (B_R): {br:>10.4f} GPa\n")
        o.write(f"Voigt's Shear Modulus         (G_V): {gv:>10.4f} GPa\n")
        o.write(f"Reuss' Shear Modulus          (G_R): {gr:>10.4f} GPa\n")
        o.write(f"Shear Modulus                 (G)  : {G:>10.4f} GPa\n")
        o.write(f"Young's Modulus               (E)  : {E:>10.4f} GPa\n")
        o.write(f"Poisson's Ratio               (v)  : {v:>10.4f}\n")
        o.write(f"P-wave Modulus                (M)  : {M:>10.4f} GPa\n")
        o.write(f"Lam\u00e9's first parameter        (L)  : {L:>10.4f} GPa\n")
        o.write(f"Pugh's Ratio                  (P)  : {P:>10.4f}\n")
        o.write(f"Sound velocity                (v_m): {vm:>10.4f} m/s\n")
        o.write(f"Debye temperature             (\u03B8_D): {tD:>10.4f} K\n")
    
    with open('Anisotropy.dat', 'w') as o:
        o.write("# Anisotropic analysis\n")
        o.write(f"Bulk Anisotropy               (A_B): {A_B:>10.4f}\n")
        o.write(f"Shear Anisotropy              (A_G): {A_G:>10.4f}\n")
        o.write(f"Universal Anisotropy          (A_U): {A_U:>10.4f}\n")
        o.write(f"(100) Planar Shear Anisotropy (A_1): {A1:>10.4f}\n")
        o.write(f"(010) Planar Shear Anisotropy (A_2): {A2:>10.4f}\n")
        o.write(f"(001) Planar Shear Anisotropy (A_3): {A3:>10.4f}\n")

def run_3d(structure, elastic_coef):
    elastic_3d = np.copy(elastic_coef)
    write_elastic_3d(elastic_3d)
    
    crystal_system = get_crystal_system(structure)
    print(f"This material is {crystal_system}")
    
    check_stability_3d(elastic_3d)
    
    props = compute_mechanical_properties_3d(elastic_3d, structure)
    print_and_write_mechanical_properties_3d(props)

def main():
    if '-h' in argv or len(argv) != 3:
        usage()
    
    structure = read_structure(argv[1])
    elastic_coef = read_elastic_tensor(argv[2])
    
    print("""Choices of type of material
1) 2D materials
2) bulk materials""")
    
    while True:
        input_type = input("Enter choice: ")
        if input_type.isdigit():
            if input_type == '1':
                run_2d(structure, elastic_coef)
                break
            elif input_type == '2':
                run_3d(structure, elastic_coef)
                break
            else:
                print("Warning! Wrong input")
        else:
            print("Warning! Wrong input")

if __name__ == '__main__':
    main()
