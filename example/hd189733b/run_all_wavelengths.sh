#!/bin/bash

rm result.txt lightcurves.txt
for i in {3940..4980..50}
do
    echo $i $(($i+50))
    python -u ../sparta/extract_eclipse_limited.py hd189733b.cfg $i $(($i+50)) --burn-in-runs 2000 --production-runs 1000 -b 1 -e 580 &
done

wait
