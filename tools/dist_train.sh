#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
GPUS=$2
export CUDA_VISIBLE_DEVICES=1,2,3,4
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
 $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
