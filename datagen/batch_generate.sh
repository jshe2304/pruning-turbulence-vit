#!/bin/bash

reynolds=(2384 570 1690 2352 4358 4212 1936 3953 2086 1744)

for r in "${reynolds[@]}"
do
    qsub -v Re="$r" /glade/u/home/jshen/pruning-turbulence-vit/datagen/generate.pbs
done
