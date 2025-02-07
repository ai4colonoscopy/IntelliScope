#!/bin/bash

EXP_MODEL_ID=cache/checkpoint/ColonGPT-phi1.5-siglip-lora-stg2 # or ColonGPT-phi1.5-siglip-stg1
LLM_PATH=cache/downloaded-weights/phi-1.5
IMAGE_FILE=cache/examples/example1.png

python colongpt/serve/cli.py \
    --model-path $EXP_MODEL_ID \
    --model-base $LLM_PATH \
    --model-type phi-1.5 \
    --image-file $IMAGE_FILE