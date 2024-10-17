#!/bin/bash

# Please change the follow configs
LLM_PATH=cache/downloaded-weights/phi-1.5 # or "microsoft/phi-1_5"
VISION_MODEL=cache/downloaded-weights/siglip-so400m-patch14-384 # or "google/siglip-so400m-patch14-384"
DATA_PATH=cache/data/ColonINST/Json-file/train/ColonINST-train.json
IMAGE_FOLDER=cache/data/ColonINST/Positive-images
OUTPUT_FILE=ColonGPT-v1-phi1.5-siglip-full
OUTPUT_DIR=cache/checkpoint/$OUTPUT_FILE

deepspeed --master_port 26000 colongpt/train/train.py \
    --deepspeed script/deepspeed_configs/zero3.json \
    --model_name_or_path $LLM_PATH \
    --model_type phi-1.5 \
    --version colongpt \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_MODEL \
    --mm_projector_type ppc_14_7_1 \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
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
    --learning_rate 2e-3 \
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