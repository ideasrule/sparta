#!/bin/bash

for filename in "$@"
do
    echo $filename
    python calibrate.py $filename
    python sub_bkd.py rateints_$filename
    python optimal_extract.py bkdsub_rateints_$filename
done
