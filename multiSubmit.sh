#!/bin/bash
OUTPUT=/home/brian.atkinson/thesis/data/noGPU/noNormDiscount
echo -e "saving experiment to:\n$OUTPUT\n"

JOB=1
for HUMAN in false
do
    for SEED in 1
    do
        for LEAKY in false
        do
            for INIT in Xavier
            do
                for L2 in false
                do
                    for L2Constant in .001
                    do
                        for LR in .0001.00001
                        do
                        sbatch --job-name=$JOB\
                        --export=ALL,HUMAN=$HUMAN,SEED=$SEED,LEAKY=$LEAKY,OUTPUT=$OUTPUT,INIT=$INIT,L2=$L2,L2Constant=$L2Constant,LR=$LR \
                        --output=/home/brian.atkinson/thesis/FlappyBird/text_files/noRMS_S$SEED.Leak$LEAKY.Init$INIT.txt \
                        mainScript.sh
                        let JOB=JOB+1
                        done
                    done
                done
            done
        done
    done
done