#!/bin/bash
set -x

cd /home/dji/projects/Vim/dep

CONFIG=configs/vim/depth/depth_vim_small_24_512_60k.py
WORK_DIR=work_dirs/vimdep-s

# Single GPU training (RTX 5080)
python train.py \
  ${CONFIG} \
  --work-dir ${WORK_DIR} \
  --launcher none \
  --gpus 1 \
  ${@:1}
