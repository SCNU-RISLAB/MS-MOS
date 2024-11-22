#!/bin/bash
export CUDA_VISIBLE_DEVICES=3 && python3 ./train_2stage.py \
                                        -d /data/datasets/kitti/ \
                                        -ac ./train_yaml/mos_pointrefine_stage.yml \
                                        -dc ./config/labels/semantic-kitti-mos.raw.yaml \
                                        -l  log/train_v2 \
                                        -p /data/zlf/mos3d/MotionRV-1stage/pretrain
                                        # -n your_projectname \
