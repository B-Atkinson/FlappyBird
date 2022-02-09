#!/bin/bash
#SBATCH --job-name=human
#SBATCH --nodes=1
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=human_%j.txt

. /etc/profile

module load lang/miniconda3 lib/cuda/11.5

source activate py38

python FB_Main.py \
--num_episodes=100000 \
--loss_reward=-5 \
--save_stats=200 \
--render=false \
--hidden=200 \
--gamma=0.99 \
--dropout=0 \
--learning_rate=0.0001 \
--seed=24 \
--decay_rate=0.99 \
--batch_size=10 \
--normalize=false \
--human=true \
--human_influence=0.4 \
--human_decay=0 \
--hidden_save_rate=200 \
--save_every=200 \
--continue_training=false \
--checkpoint_path=null