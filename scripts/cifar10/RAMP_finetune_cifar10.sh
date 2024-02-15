#!/bin/bash

# preact-resnet18
for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES='1' python RAMP.py --lr-max 0.05 --finetune_model --lr-schedule=piecewise-ft --model_name pretr_Linf --at_iter 10 --epochs 3 --final_eval --eval_freq 1 --fname Linf_finetune_kl_normalized_1.5_$seed --kl --max --seed $seed
done

# run other cifar10 bigger models

# RN-50 engstorm
python RAMP.py --lr-max 0.01  --finetune_model --lr-schedule=piecewise-ft --model_name RB_Engstrom2019Robustness --at_iter 10 --epochs 3 --final_eval --eval_freq 10 --fname Linf_finetune_cifar10_RB_Engstrom --kl --max --n_ex_final 1000

# WRN-34-20 gowal
python RAMP.py --lr-max 0.01  --finetune_model --lr-schedule=piecewise-ft --model_name RB_Gowal2020Uncovering_34_20 --at_iter 10 --epochs 3 --final_eval --eval_freq 10 --fname Linf_finetune_cifar10_RB_Gowal_34_20 --kl --max --n_ex_final 1000

# WRN-28-10 carmon
python RAMP_cifar10_aug.py --lr-max 0.01  --finetune_model --lr-schedule=piecewise-ft --model_name RB_Carmon2019Unlabeled --at_iter 10 --epochs 3 --final_eval --eval_freq 10 --fname Linf_finetune_cifar10_RB_Carmon_aug --kl --max --n_ex_final 1000

# WRN-28-10 gowal
python RAMP_cifar10_aug.py --lr-max 0.01  --finetune_model --lr-schedule=piecewise-ft --model_name RB_Gowal2020Uncovering_28_10_extra --at_iter 10 --epochs 3 --final_eval --eval_freq 10 --fname Linf_finetune_cifar10_RB_Gowal_28_10_extra_aug --kl --max --n_ex_final 1000

# WRN-70-16 gowal
python RAMP_cifar10_aug.py --lr-max 0.01  --finetune_model --lr-schedule=piecewise-ft --model_name RB_Gowal2020Uncovering_70_16_extra --at_iter 10 --epochs 3 --final_eval --eval_freq 10 --fname Linf_finetune_cifar10_RB_Gowal_70_16_extra_aug --kl --max --n_ex_final 1000 --batch_size 64