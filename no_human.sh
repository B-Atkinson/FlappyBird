#!/bin/bash
#SBATCH --job-name=NH_L0001
#SBATCH --nodes=1
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=pure_learning_.0001_%j.txt

. /etc/profile

module load lang/miniconda3 lib/cuda/11.5

source activate py38

python FB_Main.py \
--num_episodes=400000 \
--seed=42 \
--loss_reward=-5 \
--save_stats=200 \
--render=false \
--gamma=0.99 \
--learning_rate=0.0001 \
--decay_rate=0.99 \
--batch_size=200 \
--human=false \
--hidden_save_rate=400 \
--continue_training=false \
--checkpoint_path=null \
--output_dir=/home/brian.atkinson/thesis/data/Learning_0.0001
