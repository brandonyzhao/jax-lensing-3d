#!/bin/bash

for i in $(seq 64 95);
do 
    echo $i
    python train_abacus_fixps.py -d 6 -vol c009_3 -L 2 -Lz 5 -proj gg -shape_sigma 0.25 -lam 1e-3 -init_seed $i
done