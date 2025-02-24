# #!/bin/bash

python test.py \
        --data ultralytics/cfg/datasets/30_plane_rgb.yaml \
        --gpu 0 \
        --weight baseline_30shot_rgb.pt

python test.py \
        --data ultralytics/cfg/datasets/30_plane_sar.yaml \
        --gpu 0 \
        --weight baseline_30shot_sar.pt

python test.py \
        --data ultralytics/cfg/datasets/10_plane_rgb.yaml \
        --gpu 0 \
        --weight baseline_10shot_rgb.pt

python test.py \
        --data ultralytics/cfg/datasets/10_plane_sar.yaml \
        --gpu 0 \
        --weight baseline_10shot_sar.pt
