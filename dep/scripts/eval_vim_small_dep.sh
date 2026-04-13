#!/bin/bash
set -x

cd /home/dji/projects/Vim/dep

CONFIG=configs/vim/depth/depth_vim_small_24_512_60k.py
CHECKPOINT=$1
WORK_DIR=work_dirs/vimdep-s

python -m torch.distributed.launch --nproc_per_node=4 \
  test.py \
  ${CONFIG} \
  ${CHECKPOINT} \
  --work-dir ${WORK_DIR} \
  --eval AbsRel RMSE delta_1 \
  --launcher pytorch
