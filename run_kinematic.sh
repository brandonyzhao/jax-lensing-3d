#!/bin/bash

for i in $(seq 0 99);
do 
    echo $i
    python train_grid_cosmo_2.py -d 0 -vol pm0 -L 2 -Lz 5 -proj gg -gpp 64 -shape_sigma 0.035 -lam 5 -init_seed $i
done