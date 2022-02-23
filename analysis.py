import csv
from pathlib import Path
from os.path import join
import matplotlib.pyplot as plt

#plot the individual agent performances over time\
batch = 20
dir = '/home/brian.atkinson/thesis/data/Learning_0.0001/'
# targets = list(join(dir,child) for child in Path(dir).iterdir())
# print(targets)
subdir = ['ht-200000-S24-loss-5.0-hum0.4-learn0.0001','ht-200000-S42-loss-5.0-hum0.4-learn0.0001',
            'no_ht-400000-S24-loss-5.0-learn0.0001', 'no_ht-400000-S42-loss-5.0-learn0.0001']
for d in subdir:
    file = dir+d
    eps=[]
    raw_scores=[]
    raw_rewards = []
    trainer='No Human Heuristic'

    parts = d.split('-')
    for part in parts:
        if part[0]=='S':
            seed=part[1:]
        if part=='ht':
            trainer='with Human Heuristic'
    
    with open(file+'/stats.csv',newline='') as csvFile:
        reader = csv.reader(csvFile,delimiter=',')
        for line in reader:
            ep,score=int(line[0]),int(line[1])
            eps.append(ep)
            raw_scores.append(score)

    cumulative_scores = []
    running_rewards = []
    r_sum = 0   
    running_reward = 0 
    for e in range(len(eps)):
        if e % batch == 0:
            #treat batch # of games as an episode for plotting and calculating running reward
            cumulative_scores.append(r_sum)
            running_reward = .99*running_reward+.01*r_sum
            running_rewards.append(running_reward)
            r_sum = 0
        r_sum += raw_scores[e]

    length = len(cumulative_scores)
    plt.clf()
    plt.scatter(range(length),cumulative_scores,marker='.')
    plt.title('Episode Scores (Seed={}, {})'.format(seed,trainer))
    plt.xlabel('Episodes (Batch of {} games)'.format(batch))
    plt.ylabel('Number of pipes')
    plt.savefig(file+'/num_pipes.png')
    plt.clf()

    plt.scatter(range(length),running_rewards,marker='.')
    plt.title('Running Reward (Seed={}, {})'.format(seed,trainer))
    plt.xlabel('Episodes (Batch of {} games)'.format(batch))
    plt.ylabel('Running Average Score')
    plt.savefig(file+'/running_reward.png')

