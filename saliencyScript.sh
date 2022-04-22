#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --nodes=3 
#SBATCH --ntasks-per-node=10
#SBATCH --mem=16G
#SBATCH --time=00:00:20
#SBATCH --output=analysis.txt

. /etc/profile

module load lang/miniconda3

source activate py38

python asaliency.py \
--dir=/home/brian.atkinson/thesis/data/gradient_test/ht-S5-Gap1.4-Hyb1.0-FlipH_False-Leaky_True-Init_Xavier-Bias0_6785 \
--leaky=true \
--GPU=false 
