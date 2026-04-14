#!/usr/bin/env python

from sys import argv, exit
import os
import numpy as np
from ase.io import read

def usage():

    text = """
Usage: vaspPiezoelectric.py <POSCAR input> <OUTCAR input>

This script gets piezoelectric stress tensor from OUTCAR file
and calculates piezoelectric strain tensor by getting elastic coefficients.

This script was developed by Thanasee Thanasarnsurapong.
"""
    print(text)
    exit(0)

def read_structure(poscar_file):

    if not os.path.exists(poscar_file):
        print(f"ERROR!\nFile: {poscar_file} does not exist.")
        exit(0)

    return read(poscar_file)

def read_piezo_stress(outcar_file):

    if not os.path.exists(outcar_file):
        print(f"ERROR!\nFile: {outcar_file} does not exist.")
        exit(0)

    with open(outcar_file, 'r') as f:
        outcar_lines = f.readlines()

    piezostress_index = None
    for i, line in enumerate(outcar_lines):
        if 'PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)' in line:
            piezostress_index = i
            break

    if piezostress_index is None:
        print("The 'PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z' section was not found in the OUTCAR file.")
        exit(0)

    print("The 'PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z' section was found in the OUTCAR file.")
    piezostress_lines = outcar_lines[piezostress_index + 3:piezostress_index + 6]
    piezostress_vasp  = np.array([[float(x) for x in line.split()[1:]]
                                   for line in piezostress_lines])

    # Reorder from VASP convention (xx,yy,zz,xy,yz,xz) to Voigt notation (11,22,33,44,55,66)
    piezostress_coef = piezostress_vasp[:, [0, 1, 2, 4, 5, 3]]

    return outcar_lines, piezostress_coef

def read_elastic_tensor(outcar_lines):

    moduli_index = None
    for i, line in enumerate(outcar_lines):
        if 'TOTAL ELASTIC MODULI (kBar)' in line:
            moduli_index = i
            break

    if moduli_index is None:
        return None, None

    print("The 'TOTAL ELASTIC MODULI (kBar)' section was found in the OUTCAR file.")
    moduli_lines = outcar_lines[moduli_index + 3:moduli_index + 9]
    elastic_vasp = np.array([[float(x) for x in line.split()[1:]]
                              for line in moduli_lines]) / 10  # Convert kBar to GPa

    # Reorder from VASP convention (xx,yy,zz,xy,yz,xz) to Voigt notation (11,22,33,44,55,66)
    elastic_coef = elastic_vasp[:, [0, 1, 2, 4, 5, 3]][[0, 1, 2, 4, 5, 3], :]

    return moduli_index, elastic_coef

def read_elastic_from_file():

    if not os.path.exists('Elastic.dat'):
        return None

    with open('Elastic.dat', 'r') as f:
        elastic_lines = [line.strip() for line in f.readlines()
                         if not line.lstrip().startswith("#") and line.strip()]

    elastic_coef = np.array([list(map(float, line.split())) for line in elastic_lines])

    if elastic_coef.shape == (3, 3):
        print("Your material should be 2D materials.")
        return elastic_coef
    elif elastic_coef.shape == (6, 6):
        print("Your material should be bulk (3D) materials.")
        return elastic_coef
    else:
        print("Your input elastic file was probably wrong.")
        exit(0)

def read_elastic_manual():

    print("""Enter elastic tensor manually:
For 2D materials (N/m)
C11 C12 C16
C12 C22 C26
C16 C26 C66
For 3D materials (GPa)
C11 C12 C13 C14 C15 C16
C12 C22 C23 C24 C25 C26
C13 C23 C33 C34 C35 C36
C14 C24 C34 C44 C45 C46
C15 C25 C35 C45 C55 C56
C16 C26 C36 C46 C56 C66""")

    while True:
        elastic_flat = np.array(input().split(), dtype=float)
        if len(elastic_flat) == 9:
            return elastic_flat.reshape(3, 3)
        elif len(elastic_flat) == 36:
            return elastic_flat.reshape(6, 6)
        else:
            print("Error! Input must be 9 or 36 components.")

def get_elastic_tensor(outcar_lines):

    moduli_index, elastic_coef = read_elastic_tensor(outcar_lines)

    if moduli_index is not None:
        return moduli_index, elastic_coef

    print("The 'TOTAL ELASTIC MODULI (kBar)' section was not found in the OUTCAR file.")
    print("Read Elastic coefficient from another instead.")

    elastic_coef = read_elastic_from_file()

    if elastic_coef is None:
        elastic_coef = read_elastic_manual()

    return None, elastic_coef

def compute_piezo_2d(structure, piezostress_coef, elastic_coef, moduli_index):

    vector_a = structure.cell[0]
    vector_b = structure.cell[1]
    vector_c = structure.cell[2]

    vector_n  = np.cross(vector_a, vector_b) / np.linalg.norm(np.cross(vector_a, vector_b))
    factor_2d = np.abs(np.dot(vector_c, vector_n)) # Angstrom

    piezostress_2d = piezostress_coef[:, [0, 1, 5]] * factor_2d  # Convert C/m^2*Angstrom to 10^-10 C/m

    if moduli_index is not None:
        elastic_2d = elastic_coef[np.ix_([0, 1, 5], [0, 1, 5])] * factor_2d / 10  # Convert GPa*Angstrom to N/m
    elif elastic_coef.shape == (3, 3):
        elastic_2d = elastic_coef  # Already in N/m from Elastic.dat
    elif elastic_coef.shape == (6, 6):
        elastic_2d = elastic_coef[np.ix_([0, 1, 5], [0, 1, 5])]  # Slice in-plane components
    else:
        print("ERROR! Elastic tensor for 2D materials must be 3x3 or 6x6. Got shape:", elastic_coef.shape)
        exit(0)

    return piezostress_2d, elastic_2d, factor_2d

def check_stability_2d(elastic_2d, factor_2d):

    if np.all(np.linalg.eigvalsh(elastic_2d) > 1e-5 * factor_2d):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(0)

def write_elastic_2d(elastic_2d):

    C11 = elastic_2d[0, 0]; C22 = elastic_2d[1, 1]; C66 = elastic_2d[2, 2]
    C12 = elastic_2d[0, 1]; C16 = elastic_2d[0, 2]; C26 = elastic_2d[1, 2]

    header = (
        "# Elastic tensor(N/m)\n"
        "#       C11         C12         C16\n"
        "#       C12         C22         C26\n"
        "#       C16         C26         C66\n\n"
    )
    rows = (
        f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}\n"
        f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}\n"
        f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n"
    )
    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Elastic tensor(N/m)")
    print("#       C11         C12         C16")
    print("#       C12         C22         C26")
    print("#       C16         C26         C66")
    print(f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}")
    print(f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}")
    print(f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n")

def write_piezostress_2d(piezostress_2d):

    e11 = piezostress_2d[0, 0]; e12 = piezostress_2d[0, 1]; e16 = piezostress_2d[0, 2]
    e21 = piezostress_2d[1, 0]; e22 = piezostress_2d[1, 1]; e26 = piezostress_2d[1, 2]
    e31 = piezostress_2d[2, 0]; e32 = piezostress_2d[2, 1]; e36 = piezostress_2d[2, 2]

    header = (
        "# Piezoelectric Stress(10^-10 C/m)\n"
        "#       e11         e12         e16\n"
        "#       e21         e22         e26\n"
        "#       e31         e32         e36\n\n"
    )
    rows = (
        f"   {e11:>11.4f} {e12:>11.4f} {e16:>11.4f}\n"
        f"   {e21:>11.4f} {e22:>11.4f} {e26:>11.4f}\n"
        f"   {e31:>11.4f} {e32:>11.4f} {e36:>11.4f}\n"
    )
    with open('Piezoelectric_Stress.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Piezoelectric Stress(10^-10 C/m)")
    print("#       e11         e12         e16")
    print("#       e21         e22         e26")
    print("#       e31         e32         e36")
    print(f"   {e11:>11.4f} {e12:>11.4f} {e16:>11.4f}")
    print(f"   {e21:>11.4f} {e22:>11.4f} {e26:>11.4f}")
    print(f"   {e31:>11.4f} {e32:>11.4f} {e36:>11.4f}\n")

def write_piezostrain_2d(piezostrain_2d):

    d11 = piezostrain_2d[0, 0]; d12 = piezostrain_2d[0, 1]; d16 = piezostrain_2d[0, 2]
    d21 = piezostrain_2d[1, 0]; d22 = piezostrain_2d[1, 1]; d26 = piezostrain_2d[1, 2]
    d31 = piezostrain_2d[2, 0]; d32 = piezostrain_2d[2, 1]; d36 = piezostrain_2d[2, 2]

    header = (
        "# Piezoelectric Strain(pm/V)\n"
        "#       d11         d12         d16\n"
        "#       d21         d22         d26\n"
        "#       d31         d32         d36\n\n"
    )
    rows = (
        f"   {d11:>11.4f} {d12:>11.4f} {d16:>11.4f}\n"
        f"   {d21:>11.4f} {d22:>11.4f} {d26:>11.4f}\n"
        f"   {d31:>11.4f} {d32:>11.4f} {d36:>11.4f}\n"
    )
    with open('Piezoelectric_Strain.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Piezoelectric Strain(pm/V)")
    print("#       d11         d12         d16")
    print("#       d21         d22         d26")
    print("#       d31         d32         d36")
    print(f"   {d11:>11.4f} {d12:>11.4f} {d16:>11.4f}")
    print(f"   {d21:>11.4f} {d22:>11.4f} {d26:>11.4f}")
    print(f"   {d31:>11.4f} {d32:>11.4f} {d36:>11.4f}\n")

def run_2d(structure, piezostress_coef, elastic_coef, moduli_index):

    piezostress_2d, elastic_2d, factor_2d = compute_piezo_2d(structure, piezostress_coef, elastic_coef, moduli_index)

    write_elastic_2d(elastic_2d)
    write_piezostress_2d(piezostress_2d)
    check_stability_2d(elastic_2d, factor_2d)

    compliance_2d  = np.linalg.inv(elastic_2d)
    piezostrain_2d = np.dot(piezostress_2d, compliance_2d) * 1e2  # 10^-10 m/V to pm/V

    write_piezostrain_2d(piezostrain_2d)

def compute_piezo_3d(piezostress_coef, elastic_coef):

    if elastic_coef.shape != (6, 6):
        print("ERROR! Elastic tensor for 3D materials must be 6x6. Got shape:", elastic_coef.shape)
        exit(0)

    piezostress_3d = np.copy(piezostress_coef)
    elastic_3d     = np.copy(elastic_coef)

    return piezostress_3d, elastic_3d

def check_stability_3d(elastic_3d):

    if np.all(np.linalg.eigvalsh(elastic_3d) > 1e-5):
        print("This material is mechanically stable.")
    else:
        print("This material is mechanically unstable!!")
        exit(0)

def write_elastic_3d(elastic_3d):

    C = elastic_3d

    header = (
        "# Elastic tensor(GPa)\n"
        "#       C11         C12         C13         C14         C15         C16\n"
        "#       C12         C22         C23         C24         C25         C26\n"
        "#       C13         C23         C33         C34         C35         C36\n"
        "#       C14         C24         C34         C44         C45         C46\n"
        "#       C15         C25         C35         C45         C55         C56\n"
        "#       C16         C26         C36         C46         C56         C66\n\n"
    )
    rows = ""
    for i in range(6):
        rows += "   " + " ".join(f"{C[i, j]:>11.4f}" for j in range(6)) + "\n"

    with open('Elastic.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Elastic tensor(GPa)")
    print("#       C11         C12         C13         C14         C15         C16")
    print("#       C12         C22         C23         C24         C25         C26")
    print("#       C13         C23         C33         C34         C35         C36")
    print("#       C14         C24         C34         C44         C45         C46")
    print("#       C15         C25         C35         C45         C55         C56")
    print("#       C16         C26         C36         C46         C56         C66")
    for i in range(6):
        print("   " + " ".join(f"{C[i, j]:>11.4f}" for j in range(6)))
    print()

def write_piezostress_3d(piezostress_3d):

    E = piezostress_3d

    header = (
        "# Piezoelectric Stress(C/m^2)\n"
        "#       e11         e12         e13         e14         e15         e16\n"
        "#       e21         e22         e23         e24         e25         e26\n"
        "#       e31         e32         e33         e34         e35         e36\n\n"
    )
    rows = ""
    for i in range(3):
        rows += "   " + " ".join(f"{E[i, j]:>11.4f}" for j in range(6)) + "\n"

    with open('Piezoelectric_Stress.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Piezoelectric Stress(C/m^2)")
    print("#       e11         e12         e13         e14         e15         e16")
    print("#       e21         e22         e23         e24         e25         e26")
    print("#       e31         e32         e33         e34         e35         e36")
    for i in range(3):
        print("   " + " ".join(f"{E[i, j]:>11.4f}" for j in range(6)))
    print()

def write_piezostrain_3d(piezostrain_3d):

    D = piezostrain_3d

    header = (
        "# Piezoelectric Strain(pm/V)\n"
        "#       d11         d12         d13         d14         d15         d16\n"
        "#       d21         d22         d23         d24         d25         d26\n"
        "#       d31         d32         d33         d34         d35         d36\n\n"
    )
    rows = ""
    for i in range(3):
        rows += "   " + " ".join(f"{D[i, j]:>11.4f}" for j in range(6)) + "\n"

    with open('Piezoelectric_Strain.dat', 'w') as o:
        o.write(header + rows)

    print("\n# Piezoelectric Strain(pm/V)")
    print("#       d11         d12         d13         d14         d15         d16")
    print("#       d21         d22         d23         d24         d25         d26")
    print("#       d31         d32         d33         d34         d35         d36")
    for i in range(3):
        print("   " + " ".join(f"{D[i, j]:>11.4f}" for j in range(6)))
    print()

def run_3d(piezostress_coef, elastic_coef):

    piezostress_3d, elastic_3d = compute_piezo_3d(piezostress_coef, elastic_coef)

    write_elastic_3d(elastic_3d)
    write_piezostress_3d(piezostress_3d)
    check_stability_3d(elastic_3d)

    compliance_3d  = np.linalg.inv(elastic_3d)
    piezostrain_3d = np.dot(piezostress_3d, compliance_3d) * 1e3  # C/m^2 / GPa to pm/V

    write_piezostrain_3d(piezostrain_3d)

def main():

    if '-h' in argv or len(argv) != 3:
        usage()

    structure = read_structure(argv[1])
    outcar_lines, piezostress_coef = read_piezo_stress(argv[2])
    moduli_index, elastic_coef     = get_elastic_tensor(outcar_lines)

    print("""Choices of type of material
1) 2D materials
2) bulk (3D) materials""")

    while True:
        input_type = input("Enter choice: ")
        if input_type.isdigit():
            if input_type == '1':
                run_2d(structure, piezostress_coef, elastic_coef, moduli_index)
                break
            elif input_type == '2':
                run_3d(piezostress_coef, elastic_coef)
                break
            else:
                print("Warning! Wrong input")
        else:
            print("Warning! Wrong input")

if __name__ == '__main__':
    main()
