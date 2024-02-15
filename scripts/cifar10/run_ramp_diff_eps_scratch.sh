#!/bin/bash

# fine-tuning
# Linf 0 L1 1 L2 2


# Linf
for eps in 0.00784
do
    CUDA_VISIBLE_DEVICES='3' python RAMP.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/RAMP_Linf_$eps --l_eps Linf_$eps --kl --max --source 1 --target 2 --gp --lbd 5
done 

for eps in 0.01569
do
    CUDA_VISIBLE_DEVICES='3' python RAMP.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/RAMP_Linf_$eps --l_eps Linf_$eps --kl --max --source 1 --target 0 --gp --lbd 5
done

for eps in 0.04706 0.06275
do
    CUDA_VISIBLE_DEVICES='3' python RAMP.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/RAMP_Linf_$eps --l_eps Linf_$eps --kl --max --source 0 --target 1 --gp --lbd 5
done

# L1
for eps in 6 9 15
do 
    CUDA_VISIBLE_DEVICES='3' python RAMP.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/RAMP_L1_$eps --l_eps L1_$eps --kl --max --source 0 --target 1 --gp --lbd 5
done 

for eps in 18
do
    CUDA_VISIBLE_DEVICES='3' python RAMP.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/RAMP_L1_$eps --l_eps L1_$eps --kl --max --source 1 --target 0 --gp --lbd 5
done

# L2
for eps in 0.25 0.75 1.0
do
    CUDA_VISIBLE_DEVICES='3' python RAMP.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/RAMP_L2_$eps --l_eps L2_$eps --kl --max --source 0 --target 1 --gp --lbd 5
done

for eps in 1.5
do
    CUDA_VISIBLE_DEVICES='3' python RAMP.py --lr-schedule=static --at_iter 10 --epochs 80 --save_freq 20 --final_eval --eval_freq 10 --lr-max 0.05 --fname diff_eps_pretrain/RAMP_L2_$eps --l_eps L2_$eps --kl --max --source 2 --target 0 --gp --lbd 5
done
