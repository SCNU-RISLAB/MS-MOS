#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 && python3 ./infer.py \
                                                -d /path/to/dataset \
                                                -m log/train_v2/modelpath \
                                                -l log/infer_v2 \
                                                -s valid \
                                                -prf \
