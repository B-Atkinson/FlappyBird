import csv
import matplotlib.pyplot as plt

#plot the individual agent performances over time
# dir = 'heuristic_test/ht-500000-S42-loss-5.0-hum0.4/'
dir = '/home/brian.atkinson/thesis/data/Learning_0.0001/'
subdir = ['ht-200000-S42-loss-5.0-hum0.4-learn0.0001','ht-200000-S24-loss-5.0-hum0.4-learn0.0001']
for d in subdir:
    file = dir+d
    eps=[]
    raw_scores=[]
    raw_rewards = []
    
    with open(file+'/stats.csv',newline='') as csvFile:
        reader = csv.reader(csvFile,delimiter=',')
        for line in reader:
            ep,score=int(line[0]),int(line[1])
            eps.append(ep)
            raw_scores.append(score)
            raw_rewards.append(score - 5)   #account for -5 on dying

    cumulative_scores = []
    running_rewards = []
    r_sum = 0   
    running_reward = 0 
    for e in range(len(eps)):
        r_sum += raw_scores[e]
        if e % 20 == 0:
            #treat 20 games as an episode for plotting and calculating running reward
            cumulative_scores.append(r_sum)
            running_rewards.append(.99*running_reward+.01*r_sum)
            r_sum = 0

    length = len(cumulative_scores)
    plt.clf()
    plt.scatter(range(length),cumulative_scores,marker='.')
    plt.title('Score')
    plt.xlabel('Episodes')
    plt.ylabel('Number of pipes')
    plt.savefig(file+'/num_pipes.png')
    plt.clf()

    plt.scatter(range(length),running_rewards,marker='.')
    plt.title('Running Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.savefig(file+'/running_reward.png')

