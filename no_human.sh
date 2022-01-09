#!/bin/bash
#SBATCH --job-name=daniel.deridder
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --name=pure_RL
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=titans-out-%j.txt

. /etc/profile

module load lang/miniconda3/4.8.3

source activate thesis

python trainer/task.py \
--model_dir="/data/beards/CS4321/Team_DeRidderAtkinson/code/trainer/models/mnist_tests$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="fully_connected" \
--num_epochs=15 \
--batch_size=64 \
--num_classes=10 \
--eval_metrics="accuracy" \
--optimizer="SGD" \
--callback_list="checkpoint, csv_log"