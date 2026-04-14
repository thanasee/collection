#!/usr/bin/env python

from sys import argv, exit

if '-h' in argv or len(argv) != 3:
    print ("""
Usage: vaspPiezoelectric.py <POSCAR input> <OUTCAR input>

This script get piezoelectric stress tensor from OUTCAR file
and calculate piezoelectric strain tensor by get elastic coefficients.

This script was developed by Thanasee Thanasarnsurapong.
""")
    exit(0)

import os
import numpy as np
from ase.io import read
from ase.spacegroup.symmetrize import check_symmetry

# Read input POSCAR file
poscar_file = argv[1]

if not os.path.exists(poscar_file):
    print(f"""ERROR!
File: {poscar_file} does not exist.""")
    exit(0)

structure = read(poscar_file)
spacegroup = check_symmetry(structure).number

# Read input OUTCAR file
outcar_file = argv[2]

if not os.path.exists(outcar_file):
    print(f"""ERROR!
File: {outcar_file} does not exist.""")
    exit(0)

with open(outcar_file, 'r') as f:
    outcar_lines = f.readlines()

# Initialize the index for 'PIEZOELECTRIC TENSOR'
piezostress_index = None

# Search for the line containing 'PIEZOELECTRIC TENSOR'
for i, line in enumerate(outcar_lines):
    if 'PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)' in line:
        piezostress_index = i
        break

# Check if the 'length of vectors' and 'PIEZOELECTRIC TENSOR' lines were found
if piezostress_index is not None:
    print("The 'PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z' section was found in the OUTCAR file.")
    # Extract the elastic moduli values
    piezostress_lines = outcar_lines[piezostress_index + 3:piezostress_index + 6]
    piezostress_vasp = np.array([[float(x) for x in piezostress.split()[1:]]
                                 for piezostress in piezostress_lines])
    piezostress_coef = piezostress_vasp[:, [0, 1, 2, 4, 5, 3]]
else:
    print("The 'PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z' section was not found in the OUTCAR file.")
    exit(0)

# Initialize the index for 'TOTAL ELASTIC MODULI'
moduli_index = None

# Search for the line containing 'TOTAL ELASTIC MODULI'
for i, line in enumerate(outcar_lines):
    if 'TOTAL ELASTIC MODULI (kBar)' in line:
        moduli_index = i
        break

# Check if the 'length of vectors' and 'TOTAL ELASTIC MODULI' lines were found
if moduli_index is not None:
    print("The 'TOTAL ELASTIC MODULI (kBar)' section was found in the OUTCAR file")
    # Extract the elastic moduli values
    moduli_lines = outcar_lines[moduli_index + 3:moduli_index + 9]
    elastic_vasp = np.array([[float(x) for x in moduli.split()[1:]]
                             for moduli in moduli_lines]) / 10
    elastic_coef = elastic_vasp[:, [0, 1, 2, 4, 5, 3]][[0, 1, 2, 4, 5, 3], :]
else:
    print("The 'TOTAL ELASTIC MODULI (kBar)' section was not found in the OUTCAR file.")
    
    print("Read Elastic coefficient from another instead.")
    
    if os.path.exists('Elastic.dat'):
        elastic_file = 'Elastic.dat'
        
        with open(elastic_file, 'r') as f:
            elastic_lines = [line.strip() for line in f.readlines() if not line.lstrip().startswith("#") and line.strip()]
        
        elastic_coef = np.array([list(map(float, line.split())) for line in elastic_lines])
        
        if elastic_coef.shape == (3, 3):
            print("Your material should be 2D materials.")
            elastic_2d = np.copy(elastic_coef)
        elif elastic_coef.shape == (6, 6):
            print("Your material should be bulk (3D) materials.")
            elastic_3d = np.copy(elastic_coef)
        else:
            print("Your input elastic file was probably wrong.")
            exit(0)
    else:
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
            elastic_coef = np.array(input().split(), dtype=float)
            if len(elastic_coef) == 9:
                elastic_2d = elastic_coef.reshape(3, 3)
                break
            elif len(elastic_coef) == 36:
                elastic_3d = elastic_coef.reshape(6, 6)
                break
            else:
                print("Error! Input must be 9 or 36 components.")

# Choose type of material
print("""Choices of type of material
1) 2D materials
2) bulk (3D) materials""")

while True:
    input_type = input("Enter choice: ")
    if input_type.isdigit() and input_type == '1':
        vector_a = structure.cell[0]
        vector_b = structure.cell[1]
        vector_c = structure.cell[2]
        
        vector_n = np.cross(vector_a, vector_b) / np.linalg.norm(np.cross(vector_a, vector_b))
        factor_2d = np.abs(np.dot(vector_c, vector_n)) / 10 # Angstrom to nm
        piezostress_2d = piezostress_coef[:, [0, 1, 5]] * factor_2d
        
        e11 = piezostress_2d[0, 0]
        e12 = piezostress_2d[0, 1]
        e16 = piezostress_2d[0, 2]
        e21 = piezostress_2d[1, 0]
        e22 = piezostress_2d[1, 1]
        e26 = piezostress_2d[1, 2]
        e31 = piezostress_2d[2, 0]
        e32 = piezostress_2d[2, 1]
        e36 = piezostress_2d[2, 2]
        
        piezostress_file = 'Piezoelectric_Stress.dat'
        
        with open(piezostress_file, 'w') as o:
            o.write("# Piezoelectric Stress(nC/m)\n")
            o.write("#       e11         e12         e16\n")
            o.write("#       e21         e22         e26\n")
            o.write("#       e31         e32         e36\n")
            o.write("\n")
            o.write(f"   {e11:>11.4f} {e12:>11.4f} {e16:>11.4f}\n")
            o.write(f"   {e21:>11.4f} {e22:>11.4f} {e26:>11.4f}\n")
            o.write(f"   {e31:>11.4f} {e32:>11.4f} {e36:>11.4f}\n")
    
        if moduli_index is not None:
            elastic_2d = elastic_2d * factor_2d
            
            C11 = elastic_2d[0, 0]
            C22 = elastic_2d[1, 1]
            C66 = elastic_2d[2, 2]
            C12 = elastic_2d[0, 1]
            C16 = elastic_2d[0, 2]
            C26 = elastic_2d[1, 2]
            
            elastic_file = 'Elastic.dat'
            with open(elastic_file, 'w') as o:
                o.write("# Elastic tensor(N/m)\n")
                o.write("#       C11         C12         C16\n")
                o.write("#       C12         C22         C26\n")
                o.write("#       C16         C26         C66\n")
                o.write("\n")
                o.write(f"   {C11:>11.4f} {C12:>11.4f} {C16:>11.4f}\n")
                o.write(f"   {C12:>11.4f} {C22:>11.4f} {C26:>11.4f}\n")
                o.write(f"   {C16:>11.4f} {C26:>11.4f} {C66:>11.4f}\n")
        
        else:
            C11 = elastic_2d[0, 0]
            C22 = elastic_2d[1, 1]
            C66 = elastic_2d[2, 2]
            C12 = elastic_2d[0, 1]
            C16 = elastic_2d[0, 2]
            C26 = elastic_2d[1, 2]
            
        if np.all(np.linalg.eigvalsh(elastic_2d) > 1e-5):
            print("This material is mechanically stable.")
        else:
            print("This material is mechanically unstable!!")
            exit(0)
            
        compliance_2d = np.linalg.inv(elastic_2d)
        piezostrain_2d = np.dot(piezostress_2d, compliance_2d) * 1e3 # nm/V to pm/V
        
        d11 = piezostrain_2d[0, 0]
        d12 = piezostrain_2d[0, 1]
        d16 = piezostrain_2d[0, 2]
        d21 = piezostrain_2d[1, 0]
        d22 = piezostrain_2d[1, 1]
        d26 = piezostrain_2d[1, 2]
        d31 = piezostrain_2d[2, 0]
        d32 = piezostrain_2d[2, 1]
        d36 = piezostrain_2d[2, 2]
        
        piezostrain_file = 'Piezoelectric_Strain.dat'
        
        with open(piezostrain_file, 'w') as o:
            o.write("# Piezoelectric Strain(pm/V)\n")
            o.write("#       d11         d12         d16\n")
            o.write("#       d21         d22         d26\n")
            o.write("#       d31         d32         d36\n")
            o.write("\n")
            o.write(f"   {d11:>11.4f} {d12:>11.4f} {d16:>11.4f}\n")
            o.write(f"   {d21:>11.4f} {d22:>11.4f} {d26:>11.4f}\n")
            o.write(f"   {d31:>11.4f} {d32:>11.4f} {d36:>11.4f}\n")
        
        print("""
# Piezoelectric Stress(nC/m)
#       e11         e12         e16
#       e21         e22         e26
#       e31         e32         e36""" + f"""
   {e11:>11.4f} {e12:>11.4f} {e16:>11.4f}
   {e21:>11.4f} {e22:>11.4f} {e26:>11.4f}
   {e31:>11.4f} {e32:>11.4f} {e36:>11.4f}
""")
        
        print("""
# Piezoelectric Strain(pm/V)
#       d11         d12         d16
#       d21         d22         d26
#       d31         d32         d36""" + f"""
   {d11:>11.4f} {d12:>11.4f} {d16:>11.4f}
   {d21:>11.4f} {d22:>11.4f} {d26:>11.4f}
   {d31:>11.4f} {d32:>11.4f} {d36:>11.4f}
""")

        break
    
    elif input_type.isdigit() and input_type == '2':

        piezostress_3d = np.copy(piezostress_coef)
        
        e11 = piezostress_3d[0, 0]
        e12 = piezostress_3d[0, 1]
        e13 = piezostress_3d[0, 2]
        e14 = piezostress_3d[0, 3]
        e15 = piezostress_3d[0, 4]
        e16 = piezostress_3d[0, 5]
        e21 = piezostress_3d[1, 0]
        e22 = piezostress_3d[1, 1]
        e23 = piezostress_3d[1, 2]
        e24 = piezostress_3d[1, 3]
        e25 = piezostress_3d[1, 4]
        e26 = piezostress_3d[1, 5]
        e31 = piezostress_3d[2, 0]
        e32 = piezostress_3d[2, 1]
        e33 = piezostress_3d[2, 2]
        e34 = piezostress_3d[2, 3]
        e35 = piezostress_3d[2, 4]
        e36 = piezostress_3d[2, 5]
        
        piezostress_file = 'Piezoelectric_Stress.dat'
        
        with open(piezostress_file, 'w') as o:
            o.write("# Piezoelectric Stress(C/m^2)\n")
            o.write("#       e11         e12         e13         e14         e15         e16\n")
            o.write("#       e21         e22         e23         e24         e25         e26\n")
            o.write("#       e31         e32         e33         e34         e35         e36\n")
            o.write("\n")
            o.write(f"   {e11:>11.4f} {e12:>11.4f} {e13:>11.4f} {e14:>11.4f} {e15:>11.4f} {e16:>11.4f}\n")
            o.write(f"   {e21:>11.4f} {e22:>11.4f} {e23:>11.4f} {e24:>11.4f} {e25:>11.4f} {e26:>11.4f}\n")
            o.write(f"   {e31:>11.4f} {e32:>11.4f} {e33:>11.4f} {e34:>11.4f} {e35:>11.4f} {e36:>11.4f}\n")
        
        if moduli_index is not None:
            # Define the variables for elastic calculation
            elastic_3d = np.copy(elastic_coef)
            
            C11 = elastic_3d[0, 0]
            C22 = elastic_3d[1, 1]
            C33 = elastic_3d[2, 2]
            C12 = elastic_3d[0, 1]
            C13 = elastic_3d[0, 2]
            C23 = elastic_3d[1, 2]
            C44 = elastic_3d[3, 3]
            C55 = elastic_3d[4, 4]
            C66 = elastic_3d[5, 5]
            C14 = elastic_3d[0, 3]
            C15 = elastic_3d[0, 4]
            C16 = elastic_3d[0, 5]
            C24 = elastic_3d[1, 3]
            C25 = elastic_3d[1, 4]
            C26 = elastic_3d[1, 5]
            C34 = elastic_3d[2, 3]
            C35 = elastic_3d[2, 4]
            C36 = elastic_3d[2, 5]
            C45 = elastic_3d[3, 4]
            C46 = elastic_3d[3, 5]
            C56 = elastic_3d[4, 5]
                
            if np.all(np.linalg.eigvalsh(elastic_3d) > 1e-5):
                print("This material is mechanically stable.")
            else:
                print("This material is mechanically unstable!!")
                exit(0)
                    
            compliance_3d = np.linalg.inv(elastic_3d)
            
            elastic_file = 'Elastic.dat'
            with open(elastic_file, 'w') as o:
                o.write("# Elastic tensor(GPa)\n")
                o.write("#       C11         C12         C13         C14         C15         C16\n")
                o.write("#       C12         C22         C23         C24         C25         C26\n")
                o.write("#       C13         C23         C33         C34         C35         C36\n")
                o.write("#       C14         C24         C34         C44         C45         C46\n")
                o.write("#       C15         C25         C35         C45         C55         C56\n")
                o.write("#       C16         C26         C36         C46         C56         C66\n")
                o.write("\n")
                o.write(f"   {C11:>11.4f} {C12:>11.4f} {C13:>11.4f} {C14:>11.4f} {C15:>11.4f} {C16:>11.4f}\n")
                o.write(f"   {C12:>11.4f} {C22:>11.4f} {C23:>11.4f} {C24:>11.4f} {C25:>11.4f} {C26:>11.4f}\n")
                o.write(f"   {C13:>11.4f} {C23:>11.4f} {C33:>11.4f} {C34:>11.4f} {C35:>11.4f} {C36:>11.4f}\n")
                o.write(f"   {C14:>11.4f} {C24:>11.4f} {C34:>11.4f} {C44:>11.4f} {C45:>11.4f} {C46:>11.4f}\n")
                o.write(f"   {C15:>11.4f} {C25:>11.4f} {C35:>11.4f} {C45:>11.4f} {C55:>11.4f} {C56:>11.4f}\n")
                o.write(f"   {C16:>11.4f} {C26:>11.4f} {C36:>11.4f} {C46:>11.4f} {C56:>11.4f} {C66:>11.4f}\n")
        
        piezostrain_3d = np.dot(piezostress_3d, compliance_3d) * 1e3
        
        d11 = piezostrain_3d[0, 0]
        d12 = piezostrain_3d[0, 1]
        d13 = piezostrain_3d[0, 2]
        d14 = piezostrain_3d[0, 3]
        d15 = piezostrain_3d[0, 4]
        d16 = piezostrain_3d[0, 5]
        d21 = piezostrain_3d[1, 0]
        d22 = piezostrain_3d[1, 1]
        d23 = piezostrain_3d[1, 2]
        d24 = piezostrain_3d[1, 3]
        d25 = piezostrain_3d[1, 4]
        d26 = piezostrain_3d[1, 5]
        d31 = piezostrain_3d[2, 0]
        d32 = piezostrain_3d[2, 1]
        d33 = piezostrain_3d[2, 2]
        d34 = piezostrain_3d[2, 3]
        d35 = piezostrain_3d[2, 4]
        d36 = piezostrain_3d[2, 5]
        
        piezostrain_file = 'Piezoelectric_Strain.dat'
        
        with open(piezostrain_file, 'w') as o:
            o.write("# Piezoelectric Strain(pm/V)\n")
            o.write("#       d11         d12         d13         d14         d15         d16\n")
            o.write("#       d21         d22         d23         d24         d25         d26\n")
            o.write("#       d31         d32         d33         d34         d35         d36\n")
            o.write("\n")
            o.write(f"   {d11:>11.4f} {d12:>11.4f} {d13:>11.4f} {d14:>11.4f} {d15:>11.4f} {d16:>11.4f}\n")
            o.write(f"   {d21:>11.4f} {d22:>11.4f} {d23:>11.4f} {d24:>11.4f} {d25:>11.4f} {d26:>11.4f}\n")
            o.write(f"   {d31:>11.4f} {d32:>11.4f} {d33:>11.4f} {d34:>11.4f} {d35:>11.4f} {d36:>11.4f}\n")
        
        print("""
# Piezoelectric Stress(C/m^2)\n")
#       e11         e12         e13         e14         e15         e16
#       e21         e22         e23         e24         e25         e26
#       e31         e32         e33         e34         e35         e36""" + f"""
   {e11:>11.4f} {e12:>11.4f} {e13:>11.4f} {e14:>11.4f} {e15:>11.4f} {e16:>11.4f}
   {e21:>11.4f} {e22:>11.4f} {e23:>11.4f} {e24:>11.4f} {e25:>11.4f} {e26:>11.4f}
   {e31:>11.4f} {e32:>11.4f} {e33:>11.4f} {e34:>11.4f} {e35:>11.4f} {e36:>11.4f}
""")
        
        print("""
# Piezoelectric Strain(pm/V)
#       d11         d12         d13         d14         d15         d16
#       d21         d22         d23         d24         d25         d26
#       d31         d32         d33         d34         d35         d36""" + f"""
   {d11:>11.4f} {d12:>11.4f} {d13:>11.4f} {d14:>11.4f} {d15:>11.4f} {d16:>11.4f}
   {d21:>11.4f} {d22:>11.4f} {d23:>11.4f} {d24:>11.4f} {d25:>11.4f} {d26:>11.4f}
   {d31:>11.4f} {d32:>11.4f} {d33:>11.4f} {d34:>11.4f} {d35:>11.4f} {d36:>11.4f}
""")
        
        break
    else:
        print("ERROR! Wrong input")
