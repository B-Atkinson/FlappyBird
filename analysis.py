from cProfile import run
import csv
import pathlib
import os
import matplotlib.pyplot as plt

#plot the individual agent performances over time\
batch = 20
tgt = 'verification'
dataDir = pathlib.Path('../data/noGPU/'+tgt)

for dir in list(exp for exp in dataDir.iterdir()):
    experiment = os.path.split(dir)[1]
    eps=[]
    raw_scores=[]
    raw_rewards = []
    trainer='No Human Heuristic'
    elements = [trainer]
    parts = experiment.split('-')
    for part in parts:
        try:
            if part[0]=='S':
                seed=part[1:]
        except:
            ...
        if part=='ht':
            elements[0]='Human Heuristic'
        elif part[:3]=='Hyb':
            elements.append('current-{}*previous frame\n'.format(part[3:]))
        elif part[:4]=='FlipH':
            if part.split('_')[1] == 'False':
                elements.append('Normal Heuristic')
            else:
                elements.append('Flipped Heuristic')
        elif part[:4]=='Leaky':
            if part.split('_')[1] == 'False':
                elements.append('Leaky ReLu')
            else:
                elements.append('ReLu')
        elif part[:4]=='Init':
            elements.append('{} Initialization'.format(part.split('_')[1]))
        elif part[:4]=='Bias':
            try:
                elements.append('Bias Neuron Value={}'.format(part.split('_')[0][4:]))
            except:
                elements.append('Bias Neuron Value={}'.format(part[4:]))
        
        
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
    
    with open(os.path.join(dir,'digest.txt'),'w') as f:
        f.write('directory:{}\n'.format(experiment))
        f.write('max score at epoch:{}\n'.format(maximum))
        f.write('min score at epoch:{}'.format(minimum))
    length = len(cumulative_scores)
    plt.clf()
    plt.scatter(range(length),cumulative_scores,marker='.')
    plt.title('Episode Scores ({})'.format(', '.join(elements)))
    plt.xlabel('Episodes (Batch of {} games)'.format(batch))
    plt.ylabel('Number of pipes')
    plt.savefig(os.path.join(dir,'num_pipes.png'))
    plt.clf()

    plt.scatter(range(length),running_rewards,marker='.')
    plt.title('Running Reward ({})'.format(', '.join(elements)))
    plt.xlabel('Episodes (Batch of {} games)'.format(batch))
    plt.ylabel('Running Average Score')
    plt.savefig(os.path.join(dir,'running_reward.png'))
    print('done analyzing',dir)

