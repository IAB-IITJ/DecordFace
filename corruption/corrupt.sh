#!/bin/bash

# data path
# Replace `INPUT_DATA_PATH` with the path to the directory which contains all the cropped and aligned images.
# Replace `OUTPUT_DATA_PATH` with the path to the directory which corrupted images will be saved.
input_data_path=INPUT_DATA_PATH
output_data_path=OUTPUT_DATA_PATH

# run corruption
# for additional command line arguments check the argument parser of `corrupt-image-v3.py`
python corrupt-image-v3.py \
    --indir_path $input_data_path \
    --outdir_path $output_data_path \
    --verbose