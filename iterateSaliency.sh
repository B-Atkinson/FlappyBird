#!/bin/bash
OUTPUT=/home/brian.atkinson/thesis/data/Projection
echo -e "iterating through:\n$OUTPUT\n"
start=/home/brian.atkinson/thesis/FlappyBird
cd $OUTPUT

JOB=1
for d in */
do
    cd $d
    path=$PWD
    cd $start
    sbatch --job-name=$JOB\
    --export=ALL,TARGET=$path,JOB=$JOB,INTERVAL=20 \
    --output=/home/brian.atkinson/thesis/FlappyBird/text_files/weight_saliency_$JOB.txt \
    /home/brian.atkinson/thesis/FlappyBird/saliencyScript.sh
    let JOB=JOB+1
    cd $OUTPUT
done
cd $start