# FlappyBird
Agents are trained using human logic and reinforcement learning to play FlappyBird. 

The ple package is a customized version of pygames learning environment, which was sourced through GitHub. 

This GitHub repository uses branches to conduct 4 different functionalities:
-main: trains the FlappyBird agent utilizing GPUs which runs approximately 4x faster than training on CPUs, and uses the original human heuristic. 
-Projection: trains the FlappyBird agent utilizing GPUs and the projection heuristic
-noGPU: trains the FlappyBird agent utilizing CPUs (for resource-constrained training) and the original heuristic
-analysis: is used to generate all the images/saliency maps/reward curves for training; uses DFS and multiprocessing to conduct simultaneous analysis
           on all seeds belonging to an experiment, as well as all experiments in the main data directory depending on how the shell script is invoked


Main and noGPU branch training submission:
Bulk training can be done looping through multiple hyperparameter permutations using nested shell scripts contained in this repository or singularly. To 
submit a batch training job in Hamming looping through many different hyperparameters:
    1) ensure 'multisubmit.sh' and 'mainScript.sh' are executable using chmod
    1) update the 'multisubmit.sh' file
        a) OUTPUT - set to the parent directory you want to have all individual tests saved to 9(i.e. /home/brian/experiment1)
        b) create a for-loop for each hyperparameter you wish to have different values on (make sure your params.py file has a parameter that
           can take that value)
           ex: 
                for HUMAN in false true
                do
                    for SEED in 1 2 3 4
                    do
                    ...
                    done
                done
        c) update the '--export=ALL,... \' line to pass in each of the variables you created for-loops for in step 1b
           ex: 
                --export=ALL,HUMAN=$HUMAN,SEED=$SEED \
           on the left of the = sign is the variable being passed in to the job submission script, and the $VARIABLE is the value you wish passed in 
           that is set in the for-loop. Using the above example, $HUMAN is either false or true, and $SEED is 1 2 3 or 4. 
        d) update the  '--output=....txt \' line to reflect where you want the slurm output text file saved to
    2) 
        a) update the 'mainScript.sh' file to have shell variables in places where you plan to iterate different values
        b) update the amount of time you would like on the '--time=...' line
        c) update your desired resources on the '--mem=...' line
        d) specify your conda environment name on the 'source activate ...' line where the <> is
            ex:
                #!/bin/bash
                #SBATCH --mem=16G
                #SBATCH --time=DD-HH:MM:SS

                . /etc/profile

                module load lang/miniconda3

                source activate <conda_environment_name>

                python FB_Main.py \
                --num_episodes=200000 \
                --seed=$SEED \
                --loss_reward=-5 \
                --save_stats=200 \
                --gamma=0.99 \
                --learning_rate=0.0001 \
                --decay_rate=.99 \
                --batch_size=200 \
                --human=$HUMAN \
                --percent_hybrid=1 \
                --L2=false \
                --L2Constant=.01 \
                --init=Xavier \
                --leaky=false \
                --output_dir=$OUTPUT
    3) in the terminal, initiate training by entering '. multiSubmit.sh' which will run the shell script inside the current VM as opposed to creating a sub-shell to run it in
If you wish to run a single test, I suggest simply using the above steps but with only one value in the for-loops, which saves work on your end.

analysis branch submission:
If you wish to re-run the data analysis on all or a subset of the tests, simply specify the parent directory containing the tests (its children or grandchildren) you want to examine in the '--rootDirectory=...' line of 'analysisScript.sh', then after making the script executable simply run it using 'sbatch analysisScript.sh'.
If you would like to generate the saliency and feature maps for your experiments:
    1) ensure 'iterateSaliency.sh' and 'saliencyScript.sh' are executable
    2) 
        a) update the 'OUTPUT=...' line to parent directory containing the tests (its children or grandchildren) you want to examine
        b) update the INTERVAL which specifies the number of frames to skip (for time/resource saving, if time/resources don't matter set this to 1)
        c) update the 'output=...txt \' line to the location you wish to write the program output text to
        d) update the '/home/brian.atkinson/...' and 'start=...' lines to the directory containing 'saliencyScript.sh'
    3) initiate training using '. iterateSaliency.sh'
    