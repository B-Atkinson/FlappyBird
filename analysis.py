from cProfile import run
import csv
import pathlib
import os
from turtle import color
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
import multiprocessing as mp
import os

def plot_cdf(data, path, x_text, legend_loc, x_label, y_label, x_ticks, title=None, log_x=False):
    plt.clf()
    colors = {0:'b',2:'g',1:'xkcd:tangerine'}
    lower_bounds = (min(data[0]['X']),max(data[0]['X']))
    middle_bounds = (min(data[1]['X']),max(data[1]['X']))
    upper_bounds = (min(data[2]['X']),max(data[2]['X']))

    for i in range(0, len(data)):
        if log_x == True:
            plt.semilogx(data[i]['X'], data[i]['Y'], label=data[i]['key'])
        else:
            plt.plot(data[i]['X'], data[i]['Y'], label=data[i]['key'])
    plt.text(lower_bounds[0]+.01,.5, 'min: {min:.4f}\nmax: {max:.4f}'.format(min=lower_bounds[0],max=lower_bounds[1]),color=colors[0])        
    plt.text(middle_bounds[0]+.01,.6, 'min: {min:.4f}\nmax: {max:.4f}'.format(min=middle_bounds[0],max=middle_bounds[1]),color=colors[1])  
    plt.text(upper_bounds[0]+.1,.1, 'min: {min:.4f}\nmax: {max:.4f}'.format(min=upper_bounds[0],max=upper_bounds[1]),color=colors[2])  

    plt.legend(loc=legend_loc, prop={'size':11})
    plt.xlabel(x_label)
    if not x_ticks == None:
        if log_x == True:
            plt.xticks(list(x_ticks), x_text)
        else:
            plt.xticks(list(x_ticks), rotation='vertical')

    plt.ylabel(y_label)
    if not title == None:
        plt.title(title)
    
    plt.xscale('symlog')
    plt.savefig(path+'.png')
    plt.close()

def makeDict(game):
    _,numNeurons = np.shape(game)
    game = rearrange(game)
    lower,median,upper = [],[],[]
    for i in range(numNeurons):
        values = np.percentile(game[i,:],[25,50,75])
        lower.append(values[0])
        median.append(values[1])
        upper.append(values[2])
    lower=np.sort(lower)
    median=np.sort(median)
    upper=np.sort(upper)
    
    r = np.array(list(i/numNeurons for i in range(1,numNeurons+1))).reshape(numNeurons,)
    L={'Y':r,'X':lower,'key':'25th Percentile'}
    M={'Y':r,'X':median,'key':'50th Percentile'}
    U={'Y':r,'X':upper,'key':'75th Percentile'}
    
    return [L,M,U]

def rearrange(oldGame):
    frames,neurs = np.shape(oldGame)
    neurons = np.zeros((neurs,frames))
    for frame in range(frames):
        actList = list(oldGame[frame])
        for neuron in range(neurs):
            neurons[neuron, frame] = actList[neuron]
    return neurons


def main(dataDir):
    batch = 20
    print('process:',os.getpid())
    # tests = list(exp for exp in dataDir.iterdir())
    tests = [dataDir]

    for dir in tests:
        
        experiment = os.path.split(dir)[1]
        print('opening:',dir)
        eps=[]
        raw_scores=[]
        trainer='No Human Heuristic'
        elements = [trainer]
        parts = experiment.split('-')
        for part in parts:
            if part[0]=='S':
                elements.append('Seed={}'.format(part[1:]))
            if part=='ht':
                elements[0]='Human Heuristic'
            elif part[:3]=='Hyb':
                elements.append('{}% Hybrid Frame\n'.format(float(part[3:])*100))
            elif 'Leaky' in part:
                if part.split('_')[1] == 'True':
                    elements.append('Leaky ReLu')
                else:
                    elements.append('ReLu')
            elif part[:4]=='Init':
                elements.append('{} Initialization'.format(part.split('_')[1]))
            
            
        try:
            with open(os.path.join(dir,'stats.csv'),newline='') as csvFile:
                reader = csv.reader(csvFile,delimiter=',')
                bestScore = -1
                bestGame = 0
                for line in reader:
                    ep,score=int(line[0]),int(line[1])
                    eps.append(ep)
                    raw_scores.append(score)
                    if bestScore < score:
                        bestGame = ep
                print('{} done reading'.format(str(dir)))
            
        except FileNotFoundError:
            print('\n***{} can\'t file****'.format(str(dir)))
            continue

        cumulative_scores = []
        running_rewards = []
        r_sum = 0   
        running_reward = 0
        maximum = 0 
        for e in range(len(eps)):
            
            if e % batch == 0:
                #treat batch # of games as an episode for plotting and calculating running reward
                cumulative_scores.append(r_sum/20)
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
            f.write('best game:{}, best score:{}\n'.format(bestGame,bestScore))
            f.write('max running reward at epoch:{}, game:{}\n'.format(maximum,maximum*20))
            f.write('min running reward at epoch:{}, game:{}'.format(minimum, minimum*20))

        #determine which pickles to load
        if (maximum*20)%100==0:
            maxP = maximum*20
        else:
            maxP = int(math.floor(maximum*20/100)*100)
        
        if minimum*20%100==0:
            minP = minimum*20 
        else:
            minP =  int(math.floor(minimum*20/100)*100)
        try:
            #plot max and min CDFs
            with open(os.path.join(dir,'activations/{}.p'.format(maxP)),'rb') as f:
                maxAct = pickle.load(f)
            maxDict = makeDict(maxAct)
            plot_cdf(maxDict,path=os.path.join(dir,'max_CDF'),x_text='Values',log_x=True,x_ticks=None,x_label='Values',y_label='Prob',title='Max Values',legend_loc='lower right')
            with open(os.path.join(dir,'activations/{}.p'.format(minP)),'rb') as f:
                minAct = pickle.load(f)
            minDict = makeDict(minAct)
            
            plot_cdf(minDict,path=os.path.join(dir,'min_CDF'),log_x=True,x_text='Values',x_ticks=None,x_label='Values',y_label='Prob',title='Min Values',legend_loc='lower right')
        except ImportError:
            print('acivations for {} require CuPy'.format(dir))
        
        length = len(cumulative_scores)
        plt.clf()
        plt.scatter(range(length),cumulative_scores,marker='.')
        plt.title('Average Game Scores ({})'.format(', '.join(elements)))
        plt.xlabel('Episodes (Batch of {} games)'.format(batch))
        plt.ylabel('Number of Pipes')
        plt.savefig(os.path.join(dir,'num_pipes.png'))
        plt.clf()

        plt.scatter(range(length),running_rewards,marker='.')
        plt.title('Running Reward ({})'.format(', '.join(elements)))
        plt.xlabel('Episodes (Batch of {} games)'.format(batch))
        plt.ylabel('Running Average Score')
        plt.savefig(os.path.join(dir,'running_reward.png'))
        print('done analyzing',dir)

if __name__=='__main__':
    data = pathlib.Path('../data/gradient_test')
    tgtList = list(exp for exp in data.iterdir())
    pool = mp.Pool(len(tgtList))
    try:
        results = pool.map_async(main,tgtList)
        
    except Exception as e:
        print(e)
    finally:
        pool.close()
        pool.join()
        print('***all processes are done***')
