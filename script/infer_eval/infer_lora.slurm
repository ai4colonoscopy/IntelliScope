#!/bin/bash

EXP_MODEL_ID=cache/checkpoint/ColonGPT-phi1.5-siglip-lora-stg2
LLM_PATH=cache/downloaded-weights/phi-1.5
IMAGE_FOLDER=cache/data/ColonINST/Positive-images
JSON_FILE=cache/data/ColonINST/Json-file
MODEL_TYPE=phi-1.5
INFER_MODE=test # or val

mkdir -p $EXP_MODEL_ID/pred

nohup python -m colongpt.serve.infer \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type $MODEL_TYPE \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-cls.json\
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_cls.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-cls.txt 2>&1 &


nohup python -m colongpt.serve.infer \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type $MODEL_TYPE \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-reg.json \
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_reg.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-reg.txt 2>&1 &


nohup python -m colongpt.serve.infer \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type $MODEL_TYPE \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-rec.json \
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_rec.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-rec.txt 2>&1 &