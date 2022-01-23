#!/bin/bash
#SBATCH --job-name=birds_arent_real
#SBATCH --nodes=1
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=human_output.txt

. /etc/profile

module load lang/miniconda3 lib/cuda/11.5

source activate py38






python FB_Main.py \
--num_episodes=100000 \
--save_stats=100 \
--render=false \
--hidden=200 \
--gamma=0.99 \
--dropout=0 \
--learning_rate=0.0001 \
--seed=42 \
--decay_rate=0.99 \
--batch_size=10 \
--normalize=false \
--human=true \
--human_influence=0.5 \
--human_decay=0 \
--num_episodes=100000 \
--save_stats=10 \
--hidden_save_rate=200 \
--save_every=200 \
--continue_training=false \
--checkpoint_path=null \
--weight_dir=null \
--graph_dir=null \
--mgpu_run=false \
--n_gpus=1 \
--use_multiprocessing=true \
--workers=6