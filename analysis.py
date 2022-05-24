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
# import cupy as cp

def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run analysis for FlappyBird reinforcement learning with human influence.')    
    parser.add_argument('--rootDirectory', type=str, default='/home/brian.atkinson/thesis/data',
                        help='The root of the experiment tree containing all experimental directories to be analyzed. The default is that ')        
    return parser.parse_args()

def isParent(dir):
    '''The name isParent is slightly misleading, as it determines if the inputted 
       directory is the parent to the individual test folders. Each test is con-
       sidered a child in this function.'''
    path = pathlib.Path(dir) if not isinstance(dir,pathlib.PosixPath) else dir
    suffixes = ['.csv','.png','.txt','.p']
    for folder in path.iterdir():
        for subfolder in folder.iterdir():
            suf = subfolder.suffix
            if suf in suffixes: 
                return True
    return False

def isTest(dir):
    '''Determines if the directory is a test based on file types contained.'''
    path = pathlib.Path(dir) if not isinstance(dir,pathlib.PosixPath) else dir
    suffixes = ['.csv','.png','.txt','.p']
    for folder in path.iterdir():
        suf = folder.suffix
        if suf in suffixes: 
            return True
    return False

def getChildren(dir):
    '''Returns a list of pathlib.PosixPaths to the children of the inputted directory.'''
    path = pathlib.Path(dir) if not isinstance(dir,pathlib.PosixPath) else dir
    return list(child for child in path.iterdir())

def DFS_tree(dirPath):
    '''Searches whatever directory is inputted to find all the individual tests done down the tree.
       Returns a list of pathlib.PosixPaths to each test across all experiments. Implements Depth-
       First Search to find all tests.'''
    dir = pathlib.Path(dirPath) if not isinstance(dirPath,pathlib.PosixPath) else dirPath
    if isTest(dir):
        #base case where the directory is a test, return itself
        return [dir]
    results = []
    for subfolder in dir.iterdir():
        if isParent(subfolder):
            #current directory is the grandparent of the tests
            results.extend(getChildren(subfolder))
        else:
            #tests are more than 2 levels down, keep searching
            results.extend(DFS_tree(subfolder))
    return results

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


def main(path):
    dataDir = pathlib.Path(path) if not isinstance(path,pathlib.PosixPath) else path
    
    #if reach here, dataDir is the parent directory of the tests
    batch = 20
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
            for layer in ['W1','W2']:
                for processed in ['raw','RMS']:
                    with open(os.path.join(dir,'{}_{}_magnitudes.txt'.format(layer,processed)),'r') as file:
                        gradients = file.readlines()                
                    plt.clf()
                    plt.plot(np.asarray(gradients,float))
                    plt.title('{} Gradient Magnitude {} RMS'.format(layer,'Before' if processed=='raw' else 'After'))
                    plt.savefig(os.path.join(dir,'{}_gradient_{}.png'.format(layer,'Before' if processed=='raw' else 'After')))
        except OSError:
            pass
        except Exception as e:
            print('\n\n***non-errno error:\n',e)            
            
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
                        bestScore = score
            
        except FileNotFoundError:
            print('\n***{} can\'t find stats file***'.format(str(dir)))

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
                    print('\n***file:{} test {} length disagreement***'.format(dir,len(running_rewards)))
                if len(running_rewards)>=2 and running_rewards[-1]<running_rewards[-2]:
                    minimum = len(running_rewards)
                if running_rewards[-1] > running_rewards[maximum]:
                    maximum = len(running_rewards)
            r_sum += raw_scores[e]
        
        with open(os.path.join(dir,'digest.txt'),'w') as f:
            f.write('directory:{}\n'.format(experiment))
            f.write('best game:{}, best score:{}\n'.format(bestGame,bestScore))
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

        needMaxActivation,loops = True,1
        while needMaxActivation and loops<5:
            try:
                #plot max and min CDFs
                with open(os.path.join(dir,'activations/{}.p'.format(maxP)),'rb') as f:
                    maxAct = pickle.load(f)
                # if isinstance(maxAct,np.ndarray):
                #     maxAct = maxAct.get()
                maxDict = makeDict(maxAct)
                plot_cdf(maxDict,path=os.path.join(dir,'max_CDF'),x_text='Values',log_x=True,x_ticks=None,x_label='Values',y_label='Prob',title='Max Values',legend_loc='lower right')
                needMaxActivation = False
            except ImportError:
                print('\n***activations for {} require CuPy***'.format(dir))
                needMaxActivation = False
            except OSError:
                if maxP%100!=0:
                    maxP = int(math.floor(maxP/100)*100)
                elif maxP%200!=0:
                    maxP -= 100
                else:
                    pass
            except Exception as e:
                print('\n***error bulding CDFs for {}***'.format(dir))
                print(e)
                needMaxActivation = False
            finally:
                loops += 1
        if needMaxActivation:
            print('\n***file not found, no activation set for {}***'.format(dir))
        
        needMinActivation,loops = True,1
        while needMinActivation and loops<5:
            try:
                with open(os.path.join(dir,'activations/{}.p'.format(minP)),'rb') as f:
                    minAct = pickle.load(f)
                minDict = makeDict(minAct)            
                plot_cdf(minDict,path=os.path.join(dir,'init_CDF'),log_x=True,x_text='Values',x_ticks=None,x_label='Values',y_label='Prob',title='Min Values',legend_loc='lower right')
                needMinActivation = False
            except ImportError:
                print('\n***activations for {} require CuPy***'.format(dir))
                needMinActivation = False
            except OSError:
                if minP%100!=0:
                    minP = int(math.floor(minP/100)*100)
                elif minP%200!=0:
                    minP -= 100
                else:
                    pass
            except Exception as e:
                print('\n***error bulding CDFs for {}***'.format(dir))
                print(e,'\n')
                needMinActivation = False
        if needMinActivation:
            print('\n***file not found, no activation set for {}***'.format(dir))

        try:
            with open(os.path.join(dir,'activations/{}.p'.format(200)),'rb') as f:
                initAct = pickle.load(f)
            initDict = makeDict(initAct)            
            plot_cdf(initDict,path=os.path.join(dir,'init_CDF'),log_x=True,x_text='Values',x_ticks=None,x_label='Values',y_label='Prob',title='Initial Values',legend_loc='lower right')
        except ImportError:
            print('\n***activations for {} require CuPy***'.format(dir))
        except OSError:
            print('\n***optimal activation file not found, using different activation set for {}***'.format(dir))
            #subtract 100 from activation file
        except Exception as e:
            print('\n***error bulding CDFs for {}***'.format(dir))
            print(e,'\n')

            
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
        plt.clf()

        print('done analyzing',dir,flush=True)
        
        return

if __name__=='__main__':
    '''Leverages multi-processing in Hamming to be able to conduct graphical and numerical
       analysis on an arbitrary number of  tests nearly simultaneously. This script will 
       first locate each individual test in the data directory tree, then create a process 
       to analyze the tests with a 1:1 process to test ratio.'''
    params = make_argparser()
    data = pathlib.Path(params.rootDirectory)
    tgtList = DFS_tree(data)
    pool = mp.Pool(len(tgtList))
    try:
        results = pool.map_async(main,tgtList)  
    except Exception as e:
        print(e)
    finally:
        pool.close()
        pool.join()
        print('\n***all processes of {} are done***'.format(str(data)))
            


