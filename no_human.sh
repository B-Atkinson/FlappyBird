#!/bin/bash
#SBATCH --job-name=no_human
#SBATCH --nodes=1
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=pure_learning_.01_%j.txt

. /etc/profile

module load lang/miniconda3 lib/cuda/11.5

source activate py38

python FB_Main.py \
--num_episodes=200000 \
--seed=42 \
--loss_reward=-5 \
--save_stats=200 \
--render=false \
--gamma=0.99 \
--learning_rate=0.01 \
--decay_rate=0.99 \
--batch_size=200 \
--human=false \
--hidden_save_rate=400 \
--continue_training=false \
--checkpoint_path=null \
--output_dir=/home/brian.atkinson/thesis/data/Learning_0.01
