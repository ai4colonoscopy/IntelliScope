#!/bin/bash

EXP_MODEL_ID=cache/checkpoint/ColonGPT-phi1.5-siglip-lora-stg2
EVAL_MODE=test # or val

python script/multimodal_benchmark/multimodal_evaluator.py \
	--json_file $EXP_MODEL_ID/pred/${EVAL_MODE}_pred_cls.json \
	--eval_task CLS \
	--eval_mode $EVAL_MODE
	
python script/multimodal_benchmark/multimodal_evaluator.py \
	--json_file $EXP_MODEL_ID/pred/${EVAL_MODE}_pred_reg.json \
	--eval_task REG \
	--eval_mode $EVAL_MODE

python script/multimodal_benchmark/multimodal_evaluator.py \
	--json_file $EXP_MODEL_ID/pred/${EVAL_MODE}_pred_rec.json \
	--eval_task REC \
	--eval_mode $EVAL_MODE