#!/usr/bin/bash

#SBATCH -J phakir_transformer_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH -p batch_grad
#SBATCH -w ariel-g4
#SBATCH -t 3-0
#SBATCH -o logs/%A.out

source /data/uwrgoy7584/init.sh
conda activate video-mamaba-suite

python main.py --dataset "phakir" --lr 0.0005 --drop_path_rate 0.15 --num_layers 8 --channel_mask_rate 0.35 --num_epochs 120 --action train
# python main.py --dataset "phakir" --lr 0.0005 --drop_path_rate 0.15 --num_layers 8 --channel_mask_rate 0.35 --num_epochs 100 --action predict --mamba --load_epoch 100