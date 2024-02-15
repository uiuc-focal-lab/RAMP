#!/bin/bash

# training
for eps in 0.00784 0.01569 0.04706 0.06275
do
    CUDA_VISIBLE_DEVICES='1' python pretrain.py --lr-max 0.05 --epochs 80 --eval_freq 10 --at_iter 10 --final_eval --fname Linf_AT_$eps --lr-schedule 'static' --norm_idx 0 --l_eps Linf_$eps 
done

for eps in 6 9 15 18
do
    CUDA_VISIBLE_DEVICES='2' python pretrain.py --lr-max 0.05 --epochs 80 --eval_freq 10 --at_iter 10 --final_eval --fname L1_AT_$eps --lr-schedule 'static' --l_eps L1 $eps --norm_idx 1
done

for eps in 0.25 0.75 1.0
do
    CUDA_VISIBLE_DEVICES='3' python pretrain.py --lr-max 0.05 --epochs 80 --eval_freq 10 --at_iter 10 --final_eval --fname L2_AT_$eps --lr-schedule 'static' --l_eps L2 $eps --norm_idx 2
done

# evaluation
for eps in 0.00784 0.01569 0.04706 0.06275
do
    CUDA_VISIBLE_DEVICES='1' python eval.py --model_name pretr_L1 --l_eps Linf_$eps --save_dir trained_models/Linf_$eps
    CUDA_VISIBLE_DEVICES='1' python eval.py --model_name pretr_L2 --l_eps Linf_$eps --save_dir trained_models/Linf_$eps
done



