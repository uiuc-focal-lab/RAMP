#!/bin/bash
nvidia-smi

# imagenet
# resnet-50
CUDA_VISIBLE_DEVICES='3' python RAMP_imagenet.py --lr-max 0.005 --finetune_model --lr-schedule=piecewise-imagenet --model_name RB_Engstrom2019Robustness --at_iter 10 --epochs 1 --final_eval  --fname 'RAMP_imagenet' --kl --max --data_dir /share/datasets/imagenet/ --lbd 3

# Salman2020Do_R50
# XCiT-S12
CUDA_VISIBLE_DEVICES='2' python RAMP_imagenet.py --lr-max 0.0001 --finetune_model --lr-schedule=piecewise-ft --model_name RB_Debenedetti2022Light_XCiT-S12 --at_iter 10 --epochs 1 --final_eval --fname 'RAMP_transformer_imagenet' --kl --max --dataset imagenet --data_dir /share/datasets/imagenet  --batch_size 64