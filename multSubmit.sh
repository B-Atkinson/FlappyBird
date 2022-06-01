#!/bin/bash
OUTPUT=/home/brian.atkinson/thesis/data/Projection
echo -e "saving experiment to:\n$OUTPUT\n"

JOB=1
for HUMAN in false true
do
    for SEED in 1 2
    do
        for LEAKY in false
        do
            for INIT in Xavier
            do
                for L2 in false
                do
                    for L2Constant in .01
                    do    
                        sbatch --job-name=$JOB\
                        --export=ALL,HUMAN=$HUMAN,SEED=$SEED,LEAKY=false,OUTPUT=$OUTPUT,INIT=$INIT,L2=$L2,L2Constant=$L2Constant \
                        --output=/home/brian.atkinson/thesis/FlappyBird/text_files/H$HUMAN.S$SEED.L$LEAKY.init$INIT.L2$L2.$L2Constant.txt \
                        mainScript.sh
                        let JOB=JOB+1
                    done           
                done
            done
        done
    done
done