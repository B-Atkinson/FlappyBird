#!/bin/bash
#SBATCH --job-name=saliency
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=4
#SBATCH --time=02:00:00
#SBATCH --output=saliency.txt

. /etc/profile
module load lang/miniconda3
source activate py38
echo "">saliency.txt

python saliency.py \
--dir=/home/brian.atkinson/thesis/data/gradient_test/ht-S5-Gap1.4-Hyb1.0-FlipH_False-Leaky_True-Init_Xavier-Bias0_6785 \
--leaky=true \
--GPU=true  \
--num_processors=4 \
--mp=true \
--interval=10
