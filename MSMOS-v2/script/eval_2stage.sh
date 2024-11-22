#!/bin/bash
python3 ./utils/evaluate_mos.py \
        -d /data/datasets/dataset/kitti_dataset \
        -p test \
        -dc config/labels/semantic-kitti-mos.raw.yaml \
        -s valid
