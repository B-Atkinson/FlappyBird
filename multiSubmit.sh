#!/bin/bash
OUTPUT=/home/brian.atkinson/thesis/data/noGPU/L2_norm
echo -e "saving experiment to:\n$OUTPUT\n"

JOB=1
for HUMAN in false
do
    for SEED in 1 #2
    do
        for LEAKY in false #true
        do
            for INIT in Xavier
            do
                for L2 in true
                do
                    for L2Constant in .0001 #.00001 .000001
                    do    
                        sbatch --job-name=$JOB\
                        --export=ALL,HUMAN=$HUMAN,SEED=$SEED,LEAKY=$LEAKY,OUTPUT=$OUTPUT,INIT=$INIT,L2=$L2,L2Constant=$L2Constant \
                        --output=/home/brian.atkinson/thesis/FlappyBird/text_files/H$HUMAN.S$SEED.L$LEAKY.init$INIT.L2$L2.txt \
                        mainScript.sh
                        let JOB=JOB+1
                    done
                done
            done
        done
    done
done