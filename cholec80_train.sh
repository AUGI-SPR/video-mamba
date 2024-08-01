#!/usr/bin/bash

#SBATCH -J cholec80_transformer_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH -p batch_grad
#SBATCH -w ariel-g1
#SBATCH -t 3-0
#SBATCH -o logs/%A.out

source /data/uwrgoy7584/init.sh
conda activate video-mamaba-suite

# python main.py --dataset "cholec80" --lr 0.0005 --drop_path_rate 0.1 --num_layers 8 --channel_mask_rate 0.3 --num_epochs 100 --action train 
python main.py --dataset "cholec80" --lr 0.0005 --drop_path_rate 0.1 --num_layers 8 --channel_mask_rate 0.3 --num_epochs 200 --action predict --load_epoch 86