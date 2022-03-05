from cProfile import run
import csv
import pathlib
import os
import matplotlib.pyplot as plt

#plot the individual agent performances over time\
batch = 20
tgt = 'LrgGap'
dataDir = pathlib.Path('../data/'+tgt)

for dir in list(exp for exp in dataDir.iterdir()):
    experiment = os.path.split(dir)[1]
    eps=[]
    raw_scores=[]
    raw_rewards = []
    trainer='No Human Heuristic'

    parts = experiment.split('-')
    for part in parts:
        try:
            if part[0]=='S':
                seed=part[1:]
        except:
            ...
        if part=='ht':
            trainer='with Human Heuristic'
    try:
        with open(os.path.join(dir,'stats.csv'),newline='') as csvFile:
            reader = csv.reader(csvFile,delimiter=',')
            for line in reader:
                ep,score=int(line[0]),int(line[1])
                eps.append(ep)
                raw_scores.append(score)
    except FileNotFoundError:
        continue

    cumulative_scores = []
    running_rewards = []
    r_sum = 0   
    running_reward = 0
    maximum = 0 
    for e in range(len(eps)):
        
        if e % batch == 0:
            #treat batch # of games as an episode for plotting and calculating running reward
            cumulative_scores.append(r_sum)
            running_reward = .99*running_reward+.01*r_sum/batch
            running_rewards.append(running_reward)
            r_sum = 0
            
            if running_rewards[-1] >= 4:
                print('file:{} test {} exceeded'.format(dir,len(running_rewards)))
            if len(running_rewards)>=2 and running_rewards[-1]<running_rewards[-2]:
                minimum = len(running_rewards)
            if running_rewards[-1] > running_rewards[maximum]:
                maximum = len(running_rewards)
        r_sum += raw_scores[e]
    
    print('\ndirectory:',experiment)
    print('max score at epoch:',maximum)
    print('min score at epoch:',minimum)
    length = len(cumulative_scores)
    plt.clf()
    plt.scatter(range(length),cumulative_scores,marker='.')
    plt.title('Episode Scores (Seed={}, {})'.format(seed,trainer))
    plt.xlabel('Episodes (Batch of {} games)'.format(batch))
    plt.ylabel('Number of pipes')
    plt.savefig(os.path.join(dir,'num_pipes.png'))
    plt.clf()

    plt.scatter(range(length),running_rewards,marker='.')
    plt.title('Running Reward (Seed={}, {})'.format(seed,trainer))
    plt.xlabel('Episodes (Batch of {} games)'.format(batch))
    plt.ylabel('Running Average Score')
    plt.savefig(os.path.join(dir,'running_reward.png'))

