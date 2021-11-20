# AMG:Attention Map Guided Vision Transformer Pruning
ViT pruning method based on attention map information. Including token pruning and head pruning
 


# Introduction

We prune the attention heads and tokens for compression of ViT 
Our pruning work are based on the [Pytorch implementation of Vision Transformer](https://github.com/asyml/vision-transformer-pytorch)

# Installation

Create environment:
```
conda create --name vit --file requirements.txt
conda activate vit
```

# Available Models

We provide [pytorch model weights](https://drive.google.com/drive/folders/1azgrD1P413pXLJME0PjRRU-Ez-4GWN-S?usp=sharing), the pre-trained ImageNet model are converted from original jax/flax weights and pre-trained CIFAR-100 model are fine-tuned from ImageNet model . 
You can download them and put the files under 'weights/pytorch' to use them.

We can also download the [original jax/flax weights](https://github.com/google-research/vision_transformer) for pruning
We'll convert the weights for you online.

# Datasets

Currently three datasets are supported: ImageNet2012, CIFAR10, and CIFAR100. 
To evaluate or fine-tune on these datasets, download the datasets and put them in 'data/dataset_name'. 


# Fine-Tune
```
python python src/train.py --exp-name vb16-finetune  --tensorboard  --n-gpu=1 --model-arch b16  --image-size 384 --batch-size 16 --data-dir data/ --dataset CIFAR100 --num-classes 100 --train-steps 20000 --lr 0.005  --wd 0.0001 --checkpoint-path ./weights/imagenet21k+imagenet2012_ViT-B_16.pth
```


# Evaluation for Pruned Models
Make sure you have downloaded the pretrained weights either in '.npz' format or '.pth' format  

evaluate the attn25-imagenet result
```
python src/eval.py --model-arch b16 --checkpoint-path [path/to/weight] --image-size 384 --batch-size 32 --data-dir data/ --dataset ImageNet --num-classes 1000
```
evaluate the attn25-cifar result
```
python src/eval.py --model-arch b16 --checkpoint-path [path/to/weight] --image-size 384 --batch-size 32 --data-dir data/ --dataset CIFAR100 --num-classes 100
```

# Prune the ViT Models
Prune 25% tokens for ImageNet pre-trained Model
```
python src/finetune.py --exp-name vb16-ImageNet-attn25-1-0-6 \
--n-gpu=1 --model-arch b16  --prune_rate 0.25 --iter_nums 1 --finetune_nums 0 --final_finetune 6 --prune_mode attn \
--checkpoint-path=[path\to\weight] \
--image-size 384 --batch-size 16 --data-dir data/ --dataset ImageNet --num-classes 1000  --lr 0.0001 --wd 0.0001
```

Prune 25% heads for CIFAR pre-trained Model
```
python src/finetune.py --exp-name vb16-CIFAR100-attn25-1-0-6 \
--n-gpu=1 --model-arch b16  --prune_rate 0.25 --iter_nums 1 --finetune_nums 0 --final_finetune 6 --prune_mode attn \
--checkpoint-path=[path\to\weight] \
--image-size 384 --batch-size 16 --data-dir data/ --dataset CIFAR100 --num-classes 100  --lr 0.0001 --wd 0.0001
```

# Results and Models

## Pretrained Results 
| upstream    | model    | dataset      |  pytorch acc  | model link                                                                                                                                                   |
|:------------|:---------|:-------------|--------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| imagenet21k | ViT-B_16 | imagenet2012 |     83.90     | [checkpoint](https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view?usp=sharing) |
| imagenet21k | ViT-B_32 | imagenet2012 |     81.14     | [checkpoint](https://drive.google.com/file/d/1GingK9L_VcJynTCYMc3iMvCh4WG7ScBS/view?usp=sharing) |
| imagenet21k | ViT-B_16 |   CIFAR100   |     92.41     | [checkpoint](https://drive.google.com/file/d/1YVLunKEGApaSKXZKewZz974gHt09Uwyf/view?usp=sharing) |


## Pruning Results

| Method                 | model    | dataset      |   MSA param    |     FLOPS     |  acc          | model link    |
|:-----------------------|:---------|:-------------|---------------:|--------------:|--------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
|  token 25%             | ViT-B_16 | CIFAR100     |     28.4M      |     48.4G     |  92.55        |               |
|  head 25%              | ViT-B_16 | CIFAR100     |     21.3M      |     47.4G     |  92.00        |               |
|  head 25% + token 25%  | ViT-B_16 | CIFAR100     |     21.3M      |     43.7G     |  91.53        |               |
|  token 25%             | ViT-B_16 | ImageNet     |     28.4M      |     48.4G     |  84.29        |               |
|  head 15% + token 25%  | ViT-B_32 | ImageNet     |     24.2M      |     10.6G     |  80.56        |               |


# Acknowledge
1. https://github.com/google-research/vision_transformer
2. https://github.com/asyml/vision-transformer-pytorch



