#!/bin/bash

for j in $(seq 0 9); do
  for i in $(seq 0 24); do 
    echo "vol nfw${j} | seed ${i}"
    python train_abacus.py -d 0 -vol "nfw${j}" -L 2 -Lz 5 -proj gg -shape_sigma 0.25 -lam 0.03 -init_seed "${i}"
  done
done
