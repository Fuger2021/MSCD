#!/bin/bash


HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
python -u test.py \
--cs_ratio=0.1 \
--epoch=60 \
--data_dir="data" \
--testset_name="Set11"
