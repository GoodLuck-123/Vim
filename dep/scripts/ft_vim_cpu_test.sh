#!/bin/bash
set -x

cd /home/dji/projects/Vim/dep

CONFIG=configs/vim/depth/depth_vim_tiny_24_512_60k.py
WORK_DIR=work_dirs/vimdep-cpu-test

# CPU-only training (for script validation)
python train.py \
  ${CONFIG} \
  --work-dir ${WORK_DIR} \
  --launcher none \
  --options \
    data.samples_per_gpu=1 \
    total_iters=100 \
    checkpoint_config.interval=50 \
  ${@:1}
