#!/bin/bash

for i in 0.03 0.1 0.3 1 3;
do 
    echo $i
    python train_abacus_fixps.py -d 4 -vol c009 -L 2 -Lz 5 -proj gg -shape_sigma 0.25 -lam $i -init_seed 0
done