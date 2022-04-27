#!/bin/bash

BATCH_SIZE=8
ENC_LENGTH=512
DEC_LENGTH=80
LR=1e-3
N_EPOCHS=20
WARMUP=3000
GPUs_ID=0,1

# NAME="t5_b${BATCH_SIZE}x3_e${ENC_LENGTH}_d${DEC_LENGTH}_lr${LR}_warmup${WARMUP}_only_last"
NAME="KorDial_raw"
CHECKPOINT="checkpoints/al_delex/${NAME}"
DATA_PATH="data"
LOG_NAME="KorDial_raw_af"

#TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=${GPUs_ID} python pretrain.py \
--output_dir=${CHECKPOINT} \
--backbone=KETI-AIR/ke-t5-base \
--distributed_strategy=ddp \
--log_name=${LOG_NAME} \
--train_file="${DATA_PATH}/encoded/kor_merge_id.json" \
--data_path=${DATA_PATH} \
--cached_train_data=cached/kor_merge_t5.json \
--n_workers=10 \
--gradient_accumulation_steps 1 \
--optimizer adafactor \
--train_batch_size ${BATCH_SIZE} \
--valid_batch_size ${BATCH_SIZE} \
--num_train_epochs ${N_EPOCHS} \
--learning_rate ${LR} \
--enc_max_seq ${ENC_LENGTH} \
--dec_max_seq ${DEC_LENGTH} \
--max_turn 15 \
--warmup_steps ${WARMUP}