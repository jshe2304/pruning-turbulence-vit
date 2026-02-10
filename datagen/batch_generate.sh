#!/bin/bash

reynolds=(4906 4712 2744 1275 1032 4848 1574 2119 4098 1315)

for r in "${reynolds[@]}"
do
    qsub -v Re="$r" /glade/u/home/jshen/pruning-turbulence-vit/datagen/generate.pbs
done
