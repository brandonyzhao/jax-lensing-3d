#!/bin/bash

for i in $(seq 0 99);
do 
    echo $i
    python train_grid_cosmo_2.py -d 1 -vol pm0 -L 2 -Lz 5 -proj gg -gpp 480 -shape_sigma 0.25 -lam 5 -init_seed $i
done