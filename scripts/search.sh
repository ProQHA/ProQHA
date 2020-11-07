#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0
CUR_DIR=$(cd `dirname $0`; pwd)
export PYTHONPATH=${CUR_DIR}:${PYTHONPATH}
PROJECT_PATH=$1
DATASET=$2
MISSION=$3
RESUME=$4
ARCH=$5
LAYERS=$6
STEP=25
LAMBDA=500
INTERVAL=1
RATIO=0.5
SEARCH_EPOCHS=30
LR=1e-3
nvidia-smi
cd ${PROJECT_PATH}

# Mix-Precision Model Searching
python3 main/search.py \
    --step=${STEP} --lam=${LAMBDA} --interval=${INTERVAL}   \
    --ratio=${RATIO} --search_epochs=${SEARCH_EPOCHS}       \
    --dataset=${DATASET} --arch=${ARCH}  --lr=${LR}         \
    --num_layers=${LAYERS} --k_bits=8 --gpus=0              \
    --data_dir /userhome/memory_data/imagenet --clip        \
    --mission=${MISSION} --resume=${RESUME}