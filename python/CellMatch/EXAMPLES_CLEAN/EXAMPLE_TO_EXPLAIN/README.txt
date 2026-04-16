In the current directory do:

python match_cells.py CONTCAR_GRAPHENE CONTCAR_slab_332
python generate_cell.py 1 --shift1z 7

then change to directory SMALL_CELL_44
generate the same supercells 
python match_cells.py CONTCAR_GRAPHENE CONTCAR_slab_332
but now select the supercell number 3
python generate_cell.py 3 --shift1z 7
