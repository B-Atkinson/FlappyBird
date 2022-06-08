#!/bin/bash
#SBATCH --nodes=3 
#SBATCH --ntasks-per-node=10
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/brian.atkinson/thesis/FlappyBird/text_files/analysis_%j.txt

. /etc/profile

module load lang/miniconda3

source activate py38

python analysis.py \
--rootDirectory=/home/brian.atkinson/thesis/data/noGPU