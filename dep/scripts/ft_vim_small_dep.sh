#!/bin/bash
set -x

cd /home/dji/projects/Vim/dep

CONFIG=configs/vim/depth/depth_vim_small_24_512_60k.py
WORK_DIR=work_dirs/vimdep-s

# Multi-GPU distributed training
python -m torch.distributed.launch --nproc_per_node=4 \
  train.py \
  ${CONFIG} \
  --work-dir ${WORK_DIR} \
  --launcher pytorch \
  ${@:1}


