#!/bin/bash
nvidia-smi

# imagenet
# resnet-50
CUDA_VISIBLE_DEVICES='3' python RAMP_imagenet.py --lr-max 0.01 --dataset imagenet --finetune_model --lr-schedule=piecewise-imagenet --model_name RB_Engstrom2019Robustness --at_iter 10 --epochs 1 --final_eval  --fname 'RAMP_imagenet' --kl --max --data_dir /share/datasets/imagenet/


# Salman2020Do_R50
# XCiT-S12
CUDA_VISIBLE_DEVICES='2' python RAMP.py --lr-max 0.0005 --finetune_model --lr-schedule=piecewise-ft --model_name RB_Debenedetti2022Light_XCiT-S12 --at_iter 10 --epochs 1 --final_eval --fname 'RAMP_transformer_imagenet' --kl --max --dataset imagenet --data_dir /share/datasets/imagenet  --batch_size 64


# eval transformer
CUDA_VISIBLE_DEVICES='3' python eval.py --model_name RB_Debenedetti2022Light_XCiT-S12 --dataset imagenet --load_model trained_models/RAMP_transformer/ep_1_0.pth


# eval resnet-50
CUDA_VISIBLE_DEVICES='1' python eval.py --model_name RB_Engstrom2019Robustness --dataset imagenet --load_model trained_models/RAMP_imagenet/ep_1_0.pth