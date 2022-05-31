#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=07-00:00:00

. /etc/profile

module load lang/miniconda3

source activate py38

python FB_Main.py \
--num_episodes=100000 \
--seed=$SEED \
--loss_reward=-5 \
--save_stats=200 \
--gamma=0.99 \
--learning_rate=0.0001 \
--decay_rate=0.99 \
--batch_size=5 \
--human=$HUMAN \
--percent_hybrid=1 \
--L2=$L2 \
--L2Constant=$L2Constant \
--init=$INIT \
--leaky=$LEAKY \
--output_dir=$OUTPUT
