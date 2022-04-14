#!/bin/bash
#SBATCH --job-name=baseLeakyReLu5
#SBATCH --mem=16G
#SBATCH --time=07-00:00:00
#SBATCH --output=base_LeakyReLu5.txt

. /etc/profile

module load lang/miniconda3

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
--human=false \
--hidden_save_rate=100 \
--gap_size=1.4 \
--flip_heuristic=false \
--percent_hybrid=1 \
--continue_training=false \
--init=Xavier \
--leaky=true \
--bias=0 \
--checkpoint_path=null \
--output_dir=/home/brian.atkinson/thesis/data/noGPU/activation_func