# RAMP
RAMP: Boosting Adversarial Robustness Against Multiple $l_p$ Perturbations \
*Enyi Jiang, Gagandeep Singh*\
[Arxiv](https://arxiv.org/abs/2402.06827v1)

We present RAMP, a framework that boosts multiple-norm robustness, via alleviating the tradeoffs in robustness among multiple $l_p$ perturbations and accuracy/robustness. By analyzing the tradeoffs from the lens of distribution shifts, we identify the key tradeoff pair, apply logit pairing, and leverage gradient projection methods to boost union accuracy with good accuracy/robustness/efficiency tradeoffs. Our results show that RAMP outperforms SOTA methods with better union accuracy, on a wide range of model architectures on CIFAR-10 and ImageNet.

## Code

### Installation
We recommend first creating a conda environment using the provided [environment.yml](https://github.com/uiuc-focal-lab/RAMP/blob/main/environment.yml):

`conda env create -f environment.yml`

### Training from Scratch

+ **Main Result**: The files `RAMP.py` and `RAMP_wide_resnet.py` allow us to train ResNet-18 and WideReset models with standard choices of epsilons. To reproduce the results in the paper, one can run `RAMP_scratch_cifar10.sh` in folder `scripts/cifar10`.
  
+ **Varying Epsilon Values**: We provide scripts of `run_ramp_diff_eps_scratch.sh` (RAMP), `run_max_diff_eps_scratch.sh` (MAX), and `run_eat_diff_eps_scratch.sh` (E-AT) in folder `scripts/cifar10` for running the training from scratch experiments with different choices of epsilons. 

### Robust Fine-tuning

+ To get pretrained versions of ResNet-18 models with different epsilon values, one can run `pretrain_diff_eps_Lp.sh` scripts in folder `scripts/cifar10`.

+ It is also possible to use models from the [Model Zoo](https://github.com/RobustBench/robustbench#model-zoo) of [RobustBench](https://robustbench.github.io/)
with `--model_name=RB_{}` inserting the identifier of the classifier from the Model Zoo (these are automatically downloaded). (credits to E-AT paper)

+ **Main Result**:  To reproduce the results in the paper with different model architectures, one can run `RAMP_finetune_cifar10.sh` in folder `scripts/cifar10` and `RAMP_finetune_imagenet.sh` in folder `scripts/imagenet`.

+ **Varying Epsilon Values**: We provide scripts of `run_ramp_diff_eps_finetune.sh` (RAMP), `run_max_diff_eps_finetune.sh` (MAX), and `run_eat_diff_eps_finetune.sh` (E-AT) in folder `scripts/cifar10` for running the robust fine-tuning experiments with different choices of epsilons. 

### Evaluation (from E-AT paper)
With `--final_eval` our standard evaluation (with APGD-CE and APGD-T, for a total of 10 restarts of 100 steps) is run for all threat models at the end of training.
Specifying `--eval_freq=k` a fast evaluation is run on test and training points every `k` epochs.

To evaluate a trained model one can run `eval.py` with `--model_name` as above for the pretrained model or `--model_name=/path/to/checkpoint/` for new or fine-tuned
classifiers. The corresponding architecture is loaded if the run has the automatically generated name. More details about the options for evaluation in `eval.py`.

## Credits
Parts of the code in this repo is based on
+ [https://github.com/fra31/robust-finetuning](https://github.com/fra31/robust-finetuning)

