#!/bin/bash

# data path
input_data_path=INPUT_DATA_PATH
output_data_path=OUTPUT_DATA_PATH

# run detection and alignment
python corrupt-image-v3.py \
    --indir_path $input_data_path \
    --outdir_path $output_data_path \
    --verbose