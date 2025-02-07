#!/bin/bash

# Please change the follow configs
LLM_PATH=cache/downloaded-weights/phi-1.5 # or "microsoft/phi-1_5"
VISION_MODEL=cache/downloaded-weights/siglip-so400m-patch14-384 # or "google/siglip-so400m-patch14-384"
DATA_PATH=cache/ColonINST/Json-file/train/ColonINST-train-cap.json
IMAGE_FOLDER=cache/ColonINST/Positive-images
OUTPUT_FILE=ColonGPT-phi1.5-siglip-stg1
OUTPUT_DIR=cache/checkpoint/$OUTPUT_FILE

mkdir -p $OUTPUT_DIR

deepspeed colongpt/train/train.py \
    --deepspeed script/deepspeed_configs/zero3.json \
    --model_name_or_path $LLM_PATH \
    --model_type phi-1.5 \
    --version colongpt \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_MODEL \
    --mm_projector_type ppc_14_7_1 \
    --tune_mm_mlp_adapter True \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 $OUTPUT_DIR/stdout-${OUTPUT_FILE}.txt