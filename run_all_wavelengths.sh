#!/bin/bash

rm result.txt lightcurves.txt
for i in {5000..11800..500}
do
    echo $i $(($i+500))
    python -u ../miri_pipeline/extract_phase_curve_limited.py gj1214.cfg $i $(($i+500)) --burn-in-runs 2000 --production-runs 1000 -b 1 --extra-phase-terms -e 525 &
    #python -u ../miri_pipeline/extract_transit_limited.py toi134.cfg $i $(($i+200)) --burn-in-runs 2000 --production-runs 1000 -b 16
done

wait


sort -gk1 result.txt > temp
mv temp result.txt
sort -gk1 lightcurves.txt > temp
mv temp lightcurves.txt
