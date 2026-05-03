# RWCARef: Reliability-Weighted Class-Aware Reference Prototyping for Noise-Robust Out-of-Distribution Detection

This repository contains code that extends the original CADRef implementation: "[*CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging*](https://openaccess.thecvf.com/content/CVPR2025/html/Ling_CADRef_Robust_Out-of-Distribution_Detection_via_Class-Aware_Decoupled_Relative_Feature_Leveraging_CVPR_2025_paper.html)", accepted at **CVPR 2025**. Building upon their codebase, we introduce **RWCARef / RWCADRef**, which improve robustness of reference-based OOD detection under **label-noise-corrupted training data** through reliability-weighted prototype construction.

## Getting Started

### 1. Install dependencies (virtual environment recommended)

```bash
pip install -r requirements.txt
```

### 2. Download datasets
We evaluate on small-scale OOD datasets.

- Small-scale OOD datasets:
    - In-distribution dataset: **CIFAR-10, CIFAR-100**
    - Out-of-distribution datasets: **SVHN, LSUN_crop, LSUN_resize, iSUN, Places (places365), Textures (dtd)**

- Download the datasets and put them in the `data` folder. See `utils/dataset.py` for corresponding directory names.

### 3. Pre-trained Model Preparation
For the **CIFAR-10** and **CIFAR-100** benchmarks, we provide the clean pre-trained checkpoints in the `checkpoints` folder. For experiments under label noise, replace them with corresponding pre-trained checkpoints available in the `extras` folder.

### 4. Run the code

```bash
python3 ood_eval.py --OOD_method RWCARef --rw_use_distance --rw_use_entropy --rw_beta_b 0.5 --rw_beta_h 1.0 --model densenet --gpu 0 --bs 64 --num_workers 2 --use_noisy_data --noise_ratio 0.4 --noise_type symmetric --ind_dataset cifar10 --ood_dataset svhn LSUN_crop LSUN_resize iSUN dtd places365
```
or
```bash
python3 ood_eval.py --OOD_method RWCADRef --rw_use_distance --rw_use_entropy --rw_beta_b 0.5 --rw_beta_h 1.0 --model densenet --gpu 0 --bs 64 --num_workers 2 --use_noisy_data --noise_ratio 0.4 --noise_type symmetric --ind_dataset cifar10 --ood_dataset svhn LSUN_crop LSUN_resize iSUN dtd places365
```
```
usage: ood_eval.py [-h] [--ind_dataset IND_DATASET] [--ood_dataset OOD_DATASET [OOD_DATASET ...]] [--model {resnet,vit,convnext,densenet,regnet,efficientnet,swin}] [--gpu GPU] [--num_classes NUM_CLASSES]
                   [--random_seed RANDOM_SEED] [--bs BS] [--OOD_method {MSP,ODIN,Energy,GEN,ReAct,DICE,GradNorm,MaxLogit,ASH,OptFS,VIM,Residual,CARef,CADRef,RWCARef,RWCADRef}] [--use_feature_cache USE_FEATURE_CACHE]
                   [--use_score_cache USE_SCORE_CACHE] [--cache_dir CACHE_DIR] [--result_dir RESULT_DIR] [--num_workers NUM_WORKERS] [--logit_method {Energy,MSP,MaxLogit,GEN}] [--use_noisy_data] [--noise_ratio NOISE_RATIO] [--noise_type {"symmetric", "asymmetric"}] [--noise_seed NOISE_SEED] [--rw_beta_b RW_BETA_B] [--rw_beta_h RW_BETA_H] [--rw_use_distance] [--rw_use_entropy] [--rw_print_stats]

options:
  -h, --help            show this help message and exit
  --ind_dataset IND_DATASET
                        in-distribution dataset name
  --ood_dataset OOD_DATASET [OOD_DATASET ...]
                        OOD dataset list
  --model {resnet,vit,convnext,densenet,regnet,efficientnet,swin}
                        model name
  --gpu GPU             gpu id
  --num_classes NUM_CLASSES
                        number of classes
  --random_seed RANDOM_SEED
                        random seed
  --bs BS               batch size
  --OOD_method {MSP,ODIN,Energy,GEN,ReAct,DICE,GradNorm,MaxLogit,ASH,OptFS,VIM,Residual,CARef,CADRef, RWCARef, RWCADRef}
                        OOD method name
  --use_feature_cache USE_FEATURE_CACHE
                        use feature cache
  --use_score_cache USE_SCORE_CACHE
                        use score cache
  --cache_dir CACHE_DIR
                        cache directory
  --result_dir RESULT_DIR
                        result directory
  --num_workers NUM_WORKERS
                        number of workers
  --logit_method {Energy,MSP,MaxLogit,GEN}
                        logit method for CADRef

  --use_noisy_data
                        whether to use noisy training data
  --noise_ratio NOISE_RATIO
                        ratio of symmetric label-noise in [0.0, 1.0], has no effect unless used with --use_noisy_data
  --noise_type {"symmetric", "asymmetric"}
                        noise type to apply
  --noise_seed NOISE_SEED
                        random seed for noise
  --rw_beta_b RW_BETA_B
                        beta_b
  --rw_beta_h RW_BETA_H
                        beta_h
  --rw_use_distance
                        whether to use distance for reliability calculation
  --rw_use_entropy
                        whether to use entropy for reliability calculation
  --rw_print_stats
                        whether to print additional statistics related to reliability calculation
```

### 5. Additional Notes
This project uses the original CADRef codebase, released under CC0 license. We retain attribution to the original authors and add RWCARef/RWCADRef extensions.