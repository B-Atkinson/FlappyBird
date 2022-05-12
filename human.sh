#!/bin/bash
#SBATCH --job-name=iterate
#SBATCH --nodes=1
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=07-00:00:00
#SBATCH --output=%j.txt

. /etc/profile

module load lang/miniconda3 lib/cuda/11.5

source activate py38

python FB_Main.py \
--num_episodes=400000 \
--seed=5 \
--loss_reward=-5 \
--save_stats=200 \
--render=false \
--gamma=0.99 \
--learning_rate=0.0001 \
--decay_rate=0.99 \
--batch_size=200 \
--human=true \
--human_influence=0.4 \
--hidden_save_rate=100 \
--gap_size=1.4 \
--percent_hybrid=1 \
--init=Xavier \
--leaky=true \
--output_dir=/home/brian.atkinson/thesis/data/iterate_pipes
