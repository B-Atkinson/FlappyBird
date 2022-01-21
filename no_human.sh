#!/bin/bash
#SBATCH --job-name=birds_arent_real
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --name=human
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=no_human_output.txt

. /etc/profile

module load lang/miniconda3

source activate thesis

python FB_Main.py \
--num_episodes=100000 \
--human=False 