#!/bin/bash
#SBATCH --job-name=saliency
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:00:20
#SBATCH --output=saliency.txt

. /etc/profile

module load lang/miniconda3

source activate py38

python saliency.py \
--dir=/home/brian.atkinson/thesis/data/gradient_test/ht-S5-Gap1.4-Hyb1.0-FlipH_False-Leaky_True-Init_Xavier-Bias0_6785 \
--leaky=true \
--GPU=true 
