#!/usr/bin/bash

#SBATCH -J phakir_causal_asmamba
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH -p batch_grad
#SBATCH -w ariel-g3
#SBATCH -t 3-0

# Accept parameters for prior_knowledge, num_decoders, low_penalty, and high_penalty
pk=$1
nd=$2
lp=$3
hp=$4

# Dynamically generate the output log file name
output_file="logs/${SLURM_JOB_ID}_causal_asmamba_${pk}_nd_${nd}_lp_${lp}_hp_${hp}.out"

# Redirect stdout and stderr to the dynamically generated file
exec > $output_file 2>&1

# Load the environment and activate conda
source /data/uwrgoy7584/init.sh
conda activate video-mamaba-suite

# Execute the Python script with the given parameters
python main.py --feature_extractor lovit_finetuned_video01 --prior_knowledge ${pk} --num_decoders ${nd} --low_penalty ${lp} --high_penalty ${hp} --dataset phakir --causal --mamba --action train --patience 20 
# python main.py --feature_extractor lovit_finetuned_video04 --prior_knowledge order --num_decoders 3 --low_penalty 1 --high_penalty 2 --dataset phakir --causal --mamba --action train --patience 20
