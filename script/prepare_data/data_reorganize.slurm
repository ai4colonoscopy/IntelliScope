#!/bin/bash

CSV_PATH=cache/data/data_reorganize.csv
DOWNLOAD_DATASET_PATH=cache/data/ori_dataset_download
COLONINST_PATH=cache/data/ColonINST/Positive-images

python colongpt/serve/data_reorganize.py \
--csv $CSV_PATH \
--base_original $DOWNLOAD_DATASET_PATH \
--base_target $COLONINST_PATH