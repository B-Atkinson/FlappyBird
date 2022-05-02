#!/bin/bash
#SBATCH --job-name=saliency
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=saliency.txt

. /etc/profile
module load lang/miniconda3
source activate py38
echo "">saliency.txt

python saliency.py \
--dir=/home/brian.atkinson/thesis/data/noGPU/weight_inits/ht-S1-Gap1.4-Hyb1.0-FlipH_False-Leaky_False-Init_He-Bias0.0_1658 \
--leaky=false \
--GPU=false 
