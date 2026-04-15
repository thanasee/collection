# A collection of Python scripts and patch files:
1. Python scripts
  - Extract and analyze the variables of thermal transport properties in the lattice part from the HDF5 output files of Phono3py
  - Calculate structural properties (e.g., distance) from structure files in VASP format
  - Extract and plot the variables of mechanical properties (e.g., elastic tensor) from output files in VASP format
  - Prepare structure files for various calculations with the VASP calculator
2. Patch files
  - Fix lattice matrix elements (forked from Chengcheng-Xiao/VASP_OPT_AXIS)
  - Edit the call to the dftd4 function in VASP source files (since VASP version 6.4.3 when dftd4 version 4+ installed)
  - Add an implicit solvation model into the VASP calculator (VASPsol, VASPsol++, and CP-VASP)
