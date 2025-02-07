#!/bin/bash

EXP_MODEL_ID=cache/checkpoint/ColonGPT-phi1.5-siglip-stg1
LLM_PATH=cache/downloaded-weights/phi-1.5
IMAGE_FOLDER=cache/data/ColonINST/Positive-images
JSON_FILE=cache/data/ColonINST/Json-file
INFER_MODE=test # or val test

mkdir -p $EXP_MODEL_ID/pred

nohup python -m colongpt.serve.infer \
    --max_new_tokens 512 \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type phi-1.5 \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-cap.json \
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_cap.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-cap.txt 2>&1 &