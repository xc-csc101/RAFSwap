## Region-Aware Face Swapping
This repository contains the official PyTorch implementation of the paper *Region-Aware Face Swapping* (CVPR2022).

## Installation
```
conda env create -f environment/psp_env.yaml
```

## Preparing Data
Download the CelebA-HQ to the path `Data` and create mask annotations.
```
cd scripts
python mask_npy.py
```

## Preparing pretrained model
Download pre-trained StyleGAN, IR-SE50, and StyleGAN inversion models to the path `pretrained_models`, which could be downloaded following the instructions of original pSp Github repository.

## Train
```
python scripts/train.py
```

## Inference
Download pre-trained model [swap](https://drive.google.com/file/d/1g3WXZLpQvIpJ6H3OGiMZ8OUaqAW1paSk/view?usp=sharing) to the path `checkpoints`
```
python inference_celeba.py
```


