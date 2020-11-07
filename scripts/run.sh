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
ARCHTYPE=origin
nvidia-smi
cd ${PROJECT_PATH}

echo $(Exprimets ${ARCH}_${LAYERS} | ${DATASET} | Mission ${MISSION});

## FLOAT BASELINE MODEL
python3 main/train.py \
        --arch=${ARCH} \
        --num_layers=${LAYERS} \
        --archtype=${ARCHTYPE} \
        --lr=0.001 --weight_decay=1e-4 \
        --train_batch_size=256 --k_bits=8 \
        --pre_k_bits=8 --ratio=1.0 \
        --mission=${MISSION} --gpus=0 \
        --train_epochs=300 --dataset=${DATASET} \
        --data_dir=/gdata/ImageNet2012/ \
        --job_dir=../outputs --clip \
        --resume=${RESUME}

## Quant BASELINE MODEL
ARCHTYPE = quant
python3 main/train.py \
        --arch=${ARCH} \
        --num_layers=${LAYERS} \
        --archtype=${ARCHTYPE} \
        --lr=0.001 --weight_decay=1e-4 \
        --train_batch_size=256 --k_bits=8 \
        --pre_k_bits=8 --ratio=1.0 \
        --mission=${MISSION} --gpus=0 \
        --train_epochs=300 --dataset=${DATASET} \
        --data_dir=/gdata/ImageNet2012/ \
        --job_dir=../outputs --clip \
        --resume=${RESUME}

