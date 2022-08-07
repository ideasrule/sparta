#!/bin/bash

DIR="$(dirname "${BASH_SOURCE[0]}")"
#ex: ./pipeline.sh ERS_NGTS10_2022_seg_???.fits
for filename in "$@"
do
    echo $filename
    python $DIR/calibrate.py $filename
    python $DIR/simple_extract.py rateints_$filename
done

#To do phase curve fitting:
#python extract_phase_curve.py ngts_10.cfg start_bin end_bin

#To do phase curve fitting while fixing transit and eclipse params, see run_all_wavelengths.sh
