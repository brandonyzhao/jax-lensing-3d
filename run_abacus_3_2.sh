#!/bin/bash

for i in $(seq 96 127);
do 
    echo $i
    python train_abacus_fixps.py -d 3 -vol c009_2 -L 2 -Lz 5 -proj gg -shape_sigma 0.25 -lam 1e-3 -init_seed $i
done