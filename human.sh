#!/bin/bash

. /etc/profile

module load lang/miniconda3

source activate thesis


#SBATCH --job-name=birds_arent_real
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --name=human
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=human_output.txt



python FB_Main.py \
--num_episodes=100000 \
--save_stats=100
