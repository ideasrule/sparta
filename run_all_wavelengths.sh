#!/bin/bash
for i in {9..387..6}
do
    echo $(($i+6))
    python -u extract_phase_curve_limited.py ngts_10.cfg $i $(($i+6))
done
