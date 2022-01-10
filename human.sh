#!/bin/bash

. /etc/profile

module load lang/miniconda3/4.8.3

source activate thesis

for i in 1 2 3 4

do
#SBATCH --job-name=birds_arent_real
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --name=human
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=test-%j.txt



python FB_Main.py \
--num_episodes=$i * 10

done