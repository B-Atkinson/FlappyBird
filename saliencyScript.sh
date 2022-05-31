#!/bin/bash
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=02:00:00

. /etc/profile
module load lang/miniconda3
source activate py38
echo "">saliency_$JOB.txt

python saliency.py \
--dir=$TARGET \
--leaky=false \
--GPU=false  \
--num_processors=1 \
--mp=false \
--interval=$INTERVAL
