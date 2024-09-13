#!/usr/bin/bash

#SBATCH -J phakir_causal_asformer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH -p batch_grad
#SBATCH -w ariel-g2
#SBATCH -t 3-0

# Accept parameters for prior_knowledge, num_decoders, low_penalty, and high_penalty
prior_knowledge=$1
num_decoders=$2
low_penalty=$3
high_penalty=$4

# Dynamically generate the output log file name
output_file="logs/${SLURM_JOB_ID}_causal_asformer_${prior_knowledge}_nd_${num_decoders}_lp_${low_penalty}_hp_${high_penalty}.out"

# Redirect stdout and stderr to the dynamically generated file
exec > $output_file 2>&1

# Load the environment and activate conda
source /data/uwrgoy7584/init.sh
conda activate video-mamaba-suite

# Execute the Python script with the given parameters
python main.py --dataset phakir --feature_extractor lovit_finetuned --prior_knowledge ${prior_knowledge} --causal --action train --num_decoders ${num_decoders} --low_penalty ${low_penalty} --high_penalty ${high_penalty} --patience 45
# python main.py --dataset phakir --feature_extractor lovit_finetuned --prior_knowledge memory --causal --action train --num_decoders 5 --low_penalty 1 --high_penalty 5 --patience 30
# python main.py --dataset phakir --feature_extractor lovit_finetuned --prior_knowledge order --causal --action train --num_decoders 3 --low_penalty 1 --high_penalty 3
# python main.py --dataset phakir --feature_extractor lovit_finetuned_fake --prior_knowledge order --causal --action predict --num_decoders 3 --low_penalty 1 --high_penalty 2 