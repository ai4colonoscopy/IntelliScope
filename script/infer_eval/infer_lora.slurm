#!/bin/bash

EXP_MODEL_ID=cache/checkpoint/ColonGPT-v1-phi1.5-siglip-lora
LLM_PATH=cache/downloaded-weights/phi-1.5
IMAGE_FOLDER=cache/data/ColonINST/Positive-images
JSON_FILE=cache/data/ColonINST/Json-file
INFER_MODE=test # or val

mkdir -p $EXP_MODEL_ID/pred

CUDA_VISIBLE_DEVICES=0 nohup python -m colongpt.serve.infer \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type phi-1.5 \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-cls.json\
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_cls.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-cls.txt 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python -m colongpt.serve.infer \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type phi-1.5 \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-reg.json \
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_reg.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-reg.txt 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python -m colongpt.serve.infer \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type phi-1.5 \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-rec.json \
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_rec.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-rec.txt 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python -m colongpt.serve.infer \
    --max_new_tokens 512 \
    --model_path $EXP_MODEL_ID \
    --model_base $LLM_PATH \
    --model_type phi-1.5 \
    --conv_mode colongpt \
    --image_dir $IMAGE_FOLDER \
    --json_file $JSON_FILE/${INFER_MODE}/ColonINST-${INFER_MODE}-cap.json \
    --output_path $EXP_MODEL_ID/pred/${INFER_MODE}_pred_cap.json > $EXP_MODEL_ID/pred/nohup-stdout-${INFER_MODE}-pred-cap.txt 2>&1 &