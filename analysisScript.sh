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

python analysis.py