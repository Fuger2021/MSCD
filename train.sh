#!/bin/bash


OMP_NUM_THREADS=1 \
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
--nproc_per_node=2 \
--master_port=23333 \
train.py \
--cs_ratio=0.1
