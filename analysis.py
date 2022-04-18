#!/usr/bin/env python3
from cProfile import run
import csv
from email import parser
import pathlib
import os
from turtle import color
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
import multiprocessing as mp
import os
import argparse

def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run analysis for FlappyBird reinforcement learning with human influence.')
    parser.add_argument('--singleExperiment', type=str2bool, default=False,
                        help='conduct analysis on a directory, containing directories of experiments. Ex: the children of the noGPU directory  \
                        are experiments, whose children are the individual tests. In other words, the grandchildren of noGPU are the tests to  \
                        be analyzed. The default is False where the target directory is the grandparent of the tests.')
    
    parser.add_argument('--rootDirectory', type=str, default='/home/brian.atkinson/thesis/data',
                        help='The root of the experiment tree containing all experimental directories to be analyzed. The default is that ')    
    
    return parser.parse_args()

def isParent(dir):
    path = pathlib.Path(dir)
    suffixes = ['.csv','.png','.txt','.p']
    for folder in path.iterdir():
        if folder.suffix() in suffixes: 
            return True
    return False

# you have to use str2bool
# because of an issue with argparser and bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def main(dataDir,params):
    if not isParent(dataDir):
        #recursively call the same function to multiprocess experiments in children directories of the root
        print('recursing {}'.format(str(dataDir)),flush=True)
        tgtList = list([test,params] for test in dataDir.iterdir())
        pool = mp.Pool(len(tgtList))
        try:
            results = pool.map_async(main,tgtList)           
        except Exception as e:
            print(e)
        finally:
            pool.close()
            pool.join()
            print('***all subprocesses of {} are done***'.format(str(dataDir)))
            return

    #if reach here, dataDir is the parent directory of the tests
    batch = 20
    print('process:',os.getpid())
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
                # print(part[3:], type(part[3:]))
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
                for line in reader:
                    ep,score=int(line[0]),int(line[1])
                    eps.append(ep)
                    raw_scores.append(score)
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
            f.write('max score at epoch:{}, game:{}\n'.format(maximum,maximum*20))
            f.write('min score at epoch:{}, game:{}'.format(minimum, minimum*20))

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
        
        return

if __name__=='__main__':
    
    print(1,flush=True)
    params = make_argparser()
    print(2,flush=True)
    data = pathlib.Path(params.rootDirectory)
    print(3,data,flush=True)
    tgtList = list((test,params) for test in data.iterdir())
    print(4,tgtList,flush=True)
    pool = mp.Pool(len(tgtList))
    print(5,flush=True)
    try:
        results = pool.map_async(main,tgtList)  
        print(6,flush=True)          
    except Exception as e:
        print(e)
    finally:
        pool.close()
        print(7,flush=True)
        pool.join()
        print(8,flush=True)
        print('***all processes of {} are done***'.format(str(data)))
            
