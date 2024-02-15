#!/bin/bash

# Linf
for eps in 0.00784 0.01569 0.04706 0.06275
do
    CUDA_VISIBLE_DEVICES='3' python MAX.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/MAX_Linf_$eps --l_eps Linf_$eps
done

# L1
for eps in 6 9 15 18
do
    CUDA_VISIBLE_DEVICES='3' python MAX.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/MAX_L1_$eps --l_eps L1_$eps
done

# L2
for eps in 0.25 0.75 1.0 1.5
do
    CUDA_VISIBLE_DEVICES='3' python MAX.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/MAX_L2_$eps --l_eps L2_$eps
done
