#!/bin/bash

for i in {238..0..6}
do
    echo $i $(($i+6))
    python -u ../miri_pipeline/extract_phase_curve_limited.py gj1214.cfg $i $(($i+6)) --burn-in-runs 2000 --production-runs 1000 -b 16
done
