#!/bin/bash


for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES='2' python RAMP.py --lr-max 0.05  --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 10 --eval_freq 10 --fname RAMP_beta_0.5_lbd_5_$seed --kl --max --final_eval --gp --lbd 5 --seed $seed
done

# pretrain on wideresnet
for seed in 0
do
    CUDA_VISIBLE_DEVICES='0' python RAMP_wide_resnet.py --lr-max 0.1  --lr-schedule=superconverge --at_iter 10 --epochs 30 --save_freq 10 --eval_freq 10 --fname RAMP_beta_0.5_lbd_2_wide_trades_$seed --kl --max --final_eval --gp --lbd 2 --seed $seed --wide
done