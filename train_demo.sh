# train the lego data of D-NeRF dataset
CUDA_LAUNCH_BLOCKING=1 python train.py -s ../../data/d-nerf/lego -m output/exp-demo --eval --is_blender