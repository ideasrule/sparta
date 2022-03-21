#!/bin/bash

#ex: ./pipeline.sh ERS_NGTS10_2022_seg_???.fits
for filename in "$@"
do
    echo $filename
    python calibrate.py $filename
    python sub_bkd.py rateints_$filename
    python optimal_extract.py bkdsub_rateints_$filename
done

#To do phase curve fitting:
#python extract_phase_curve.py ngts_10.cfg start_bin end_bin

#To do phase curve fitting while fixing transit and eclipse params, see run_all_wavelengths.sh
