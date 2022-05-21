import pathlib
# import pygame
# from pygame.constants import K_w
# from ple.games.flappybird import FlappyBird
# from ple import PLE
# import csv
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cv2 as cv
import seaborn as sns
import pickle
import argparse
import os
from math import floor, ceil
from copy import deepcopy


def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run analysis for FlappyBird reinforcement learning with human influence.')    
    parser.add_argument('--leaky', type=str2bool,default=False,
                        help='If true, will utilize Leaky ReLu activation function.')
    parser.add_argument('--dir', type=str,
                        help='The filepath to the test directory to be loaded.')  
    parser.add_argument('--GPU', type=str2bool,default=False,
                        help='If true, will run the code using CuPy and NumPy. This is required to analyze results from GPU-based tests.')
    parser.add_argument('--num_processors', type=int, default=1,
                        help='The number of processors to use in multiprocessing.')  
    parser.add_argument('--mp', type=str2bool, default=False,
                        help='If true, will run the code in multiple processes to decrease runtime. Requires num_processors to be >1.')
    parser.add_argument('--interval', type=int, default=10,
                        help='Defines how many frames to analyze. The default of 10 would analyze every 10th frame. It is recommended \
                        that the interval be no smaller than 10 for 100-frame sequences or larger, due to the runtime of this program.')
    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1'):
        return True
    elif v.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def findMaxModel(dir):
    '''Uses the digest.txt file inside of each test to determine what pickle file to open when loading the model. Takes as input the file path
    to the test directory, and returns the number of the best model available. If the pickle file of the exact best model is not available, it
    will return the next closest pickle file that is not greater than the best. As an example, if the true best model happened in game 1234, but 
    there is no 1234.p file available, the code will return 1200. If there is no digest.txt file, this function throws an exception.'''
    digest = os.path.join(dir,'digest.txt')
    if os.path.exists(digest):
        with open(digest) as fd:
            lines = fd.readlines()
        best = lines[1].split(',')[0].split(':')[1]
        bestPath = os.path.join(dir,'pickles/'+best+'.p')
        print('path to best model:',str(bestPath))
        try:
            with open(bestPath,'r') as f:
                pass
        except OSError:
            #if the exact model is not available, choose the next highest saved point without going over
            print('\n***True best model unavailable, choosing closest option.***\n\n')
            best = max(floor( int(best) / 100 ) * 100,1)   
            if not(best%200==0):
                best-=100
    #will throw an exception if the digest file does not exist, fail early
    return str(best)
    
def loadModel(dir):
    '''Takes as input the filepath to the test directory, and returns the best model available based on the search completed by findMaxModel. The
    returned object is a Numpy ndarray if the test was run purely on CPUs, and is a CuPy array if the test was done on GPUs.'''
    gameNumber = findMaxModel(dir)
    file = os.path.join(dir,'pickles/'+gameNumber+'.p')
    model  = pickle.load(open(file,'rb'))
    return model

def loadWorstModel(dir):
    '''Takes as input the filepath to the test directory, and returns the worst model available based on the digest file, which should be the last one. 
    The returned object is a Numpy ndarray if the test was run purely on CPUs, and is a CuPy array if the test was done on GPUs.'''
    digest = os.path.join(dir,'digest.txt')
    if os.path.exists(digest):
        with open(digest) as fd:
            lines = fd.readlines()
        worst = lines[3].split(',')[1].split(':')[1]
        worstPath = os.path.join(dir,'pickles/'+worst+'.p')
        print('path to worst model:',str(worstPath))
        if not os.path.exists(worstPath):
            #if the exact model is not available, choose the next highest saved point without going over
            print('\n***True worst model unavailable, choosing closest option.***\n\n')
            worst = max(floor( int(worst) / 100 ) * 100,1)   
            if not(worst%200==0):
                worst-=100
    #will throw an exception if the digest file does not exist, fail early
    file = os.path.join(dir,'pickles/'+str(worst)+'.p')
    model  = pickle.load(open(file,'rb'))
    return model

def loadFrames(dir):
    '''Takes as input the filepath to the test directory, and returns a homogenous python list containing the frames from the best performing episode of that test.
    Each item in the returned list is either a Numpy array (if the test was done purely on CPUs) or a CuPy array (if the test was done on GPUs). Drops the first 10
    frames because they tend to be zerod out.'''
    try:
        dir = os.path.join(dir,'bestFrames.p')
        print('loading frames from:',dir)
        frameList = pickle.load(open(dir,'rb'))
        # print('\n{} frames, each frame shape {}'.format(len(frameList),cp.shape(frameList[0])))
    except FileNotFoundError:
        print('no best frames in test, using another batch')
        frameList = pickle.load(open('/home/brian.atkinson/thesis/data/gradient_test/ht-S5-Gap1.4-Hyb1.0-FlipH_False-Leaky_True-Init_Xavier-Bias0_6785/bestFrames.p','rb'))
        print('\n{} frames, each frame shape {}'.format(len(frameList),cp.shape(frameList[0])))
    return frameList[10:]

def sigmoid(value):
    """Activation function used at the output of the neural network."""
    return 1.0 / (1.0 + np.exp(-value)) 

def policy_forward(screen_input, model,leaky):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively). Does not
    utilize a GPU or CuPy. Returns the probability of the agent flapping, as well 
    as the hidden node activations."""
    int_h = np.dot(model['W1'], screen_input)
    if leaky:
        # # Leaky ReLU 
        int_h[int_h < 0] *= .01
    else:
        # ReLU nonlinearity used to get hidden layer state
        int_h[int_h < 0] = 0      
    logp = np.dot(model['W2'], int_h)
    #probability of moving the agent up
    p = sigmoid(logp)
    return p, int_h 

def policy_forward_GPU(screen_input, model,leaky):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively). Uses CuPy
    to achieve GPU acceleration. Returns the probability of the agent flapping, as well 
    as the hidden node activations."""
    int_h = cp.dot(model['W1'], screen_input)
    if leaky:
        # # Leaky ReLU 
        int_h[int_h < 0] *= .01
    else:
        # ReLU nonlinearity used to get hidden layer state
        int_h[int_h < 0] = 0      
    logp = cp.dot(model['W2'], int_h)
    #probability of moving the agent up
    p = sigmoid(logp)
    return p, int_h

def makeMapCPU(origFrame,model,params,frameNum):
    '''Uses a frame and the network weights to calculate the score of each pixel. To begin the unaltered frame is passed through
    the network to calculate a base probability of flapping. The score of each pixel is calculated by applying a 5x5 Gaussian 
    Kernel to a single pixel, and then passing the modified frame through the network. The pixel's score is then determined by
    squaring the difference between the original and modified probabilities, and dividing by 2.
    Inputs:
    frame- a single cupy or numpy ndarray of shape (72,100) that contains the frame data
    model- a numpy or cupy ndarray that contains the model weights for the hidden and output layers
    params- the command line arguments passed in to the python script, this is used to determine if Leaky ReLu is used

    Outputs:
    scores- a numpy array of shape (72,100) representing the score each pixel received after being blurred and tested
    '''
    
    #create a blurred image to pull individual pixel values from, which saves computation
    #both branches accomplish the same task, getting the probability
    if isinstance(origFrame,cp.core.core.ndarray):
        origFrame = origFrame.get()
    frame = np.copy(origFrame)
    input = np.copy(frame).reshape(72,100)

    blurredImg = cv.GaussianBlur(input,(5,5),cv.BORDER_DEFAULT).ravel()
    orig_prob,_ = policy_forward(frame, model,params.leaky)
    
    if id(blurredImg) == id(frame):
        print('\nThere is an issue with blurring\n')

    new_prob = []
    old = np.array([0])
    for i in range(7200):
        #write the blurred pixel value to the target pixel, save the old value
        old[0] = np.copy(frame[i])
        frame[i] = np.copy(blurredImg[i])
        #get the perturbed probability
        p,_= policy_forward(frame,model,params.leaky)
        #save the result and write the original target pixel value back to the frame 
        new_prob.append(p)
        frame[i] = np.copy(old[0])
    
    #apply scoring function to the perturbed frame
    scores = list(map(lambda i: abs(orig_prob-i),new_prob))
    return np.array(scores).reshape(72,100)

def makeMapNotCPU(origFrame,model,params,frameNum):
    '''*****************
    Inputs:
    frame- a single cupy or numpy ndarray of shape (72,100) that contains the frame data
    model- a numpy or cupy ndarray that contains the model weights for the hidden and output layers
    params- the command line arguments passed in to the python script, this is used to determine if Leaky ReLu is used

    Outputs:
    scores- a numpy array of shape (72,100) representing the score each pixel received after being blurred and tested
    '''
    
    #create a blurred image to pull individual pixel values from, which saves computation
    #both branches accomplish the same task, getting the probability
    if isinstance(origFrame,cp.core.core.ndarray):
        origFrame = origFrame.get()
    frame = np.copy(origFrame)
    orig_prob,_ = policy_forward(frame, model,params.leaky)
    
    new_prob = []
    old = np.array([0])
    for i in range(7200):
        #write the blurred pixel value to the target pixel, save the old value
        old[0] = np.copy(frame[i])
        frame[i] = int(not(frame[i]))
        #get the perturbed probability
        p,_= policy_forward(frame,model,params.leaky)
        #save the result and write the original target pixel value back to the frame 
        new_prob.append(p)
        frame[i] = np.copy(old[0])
    
    #apply scoring function to the perturbed frame
    scores = list(map(lambda i: abs(orig_prob-i),new_prob))
    return np.array(scores).reshape(72,100)


def makeMapGPU(origFrame,model,params,frameNum):
    '''Uses a frame and the network weights to calculate the score of each pixel. To begin the unaltered frame is passed through
    the network to calculate a base probability of flapping. The score of each pixel is calculated by applying a 5x5 Gaussian 
    Kernel to a single pixel, and then passing the modified frame through the network. The pixel's score is then determined by
    squaring the difference between the original and modified probabilities, and dividing by 2.
    Inputs:
    frame- a single cupy or numpy ndarray of shape (72,100) that contains the frame data
    model- a numpy or cupy ndarray that contains the model weights for the hidden and output layers
    params- the command line arguments passed in to the python script, this is used to determine if Leaky ReLu is used

    Outputs:
    scores- a numpy array of shape (72,100) representing the score each pixel received after being blurred and tested
    '''
    
    #create a blurred image to pull individual pixel values from, which saves computation
    #both branches accomplish the same task, getting the probability
    frame = cp.copy(origFrame)
    input = cp.copy(frame.get())

    blurredImg = cp.asarray(cv.GaussianBlur(input.reshape(72,100),(5,5),cv.BORDER_DEFAULT)).ravel()
    orig_prob,_ = policy_forward_GPU(frame, model,params.leaky)
    orig_prob = orig_prob.get()
    new_prob = []
    old = cp.array([0])
    testList=[]
    for i in range(7200):
        #write the blurred pixel value to the target pixel, save the old value
        old[0] = cp.copy(frame[i])
        frame[i] = cp.copy(blurredImg[i])

        #get the perturbed probability
        p,_= policy_forward_GPU(frame,model,params.leaky)
        p = p.get()
        #save the result and write the original target pixel value back to the frame 
        new_prob.append(p)
        frame[i] = cp.copy(old[0])

        if 100<=i<150:
            print('pixel {}\norig prob={} pert prob={}\n\n'.format(i,orig_prob,p))
            testList.append(p)
            if i == 149: print(flush=True)
            
    
    #apply scoring function to the perturbed frame
    scores = list(map(lambda i: abs(orig_prob-i),new_prob))
    for i in range(100,150):
            p = testList.pop(0)
            match = (orig_prob-p)==(scores[i])
            # print('pixel {} scores match:{}'.format(i,match))
            if not match:
                print('pert prob={}   orig prob={}   auto score={}   hand score={}'.format(p,orig_prob,scores[i],orig_prob-p),flush=True)
    # print(flush=True)
    return np.array(scores).reshape(72,100)


def makeMapNot(origFrame,model,params,frameNum):
    '''Uses a frame and the network weights to calculate the score of each pixel. To begin the unaltered frame is passed through
    the network to calculate a base probability of flapping. The score of each pixel is calculated by applying a 5x5 Gaussian 
    Kernel to a single pixel, and then passing the modified frame through the network. The pixel's score is then determined by
    squaring the difference between the original and modified probabilities, and dividing by 2.
    Inputs:
    frame- a single cupy or numpy ndarray of shape (72,100) that contains the frame data
    model- a numpy or cupy ndarray that contains the model weights for the hidden and output layers
    params- the command line arguments passed in to the python script, this is used to determine if Leaky ReLu is used

    Outputs:
    scores- a numpy array of shape (72,100) representing the score each pixel received after being blurred and tested
    '''
    frame = cp.copy(origFrame)
    orig_prob,_ = policy_forward_GPU(frame, model,params.leaky)
    orig_prob = orig_prob.get()
    
    new_prob = []
    old = np.array([0])
    for i in range(7200):
        #negate pixel value
        old = np.copy(frame[i])
        frame[i] = int(not(frame[i]))

        #get the perturbed probability
        p,_= policy_forward_GPU(frame,model,params.leaky)
        p = p.get()
        #save the result and write the original target pixel value back to the frame 
        new_prob.append(p)
        frame[i] = cp.copy(old)
    scores = list(map(lambda i: abs(orig_prob-i),new_prob))
    return np.array(scores).reshape(72,100)

def plotMaps(framelist,start,end,interval,params,models):
    print('process {} starting range {}-{} at intervals of {}\n'.format(getpid(),start,end,interval))
    #create an input and model saliency map for each loaded frame
    for i in range(start,end,interval):
        print('\n\n*********mp frame {}**********'.format(i),flush=True)
        plt.close()
        try:
            plt.imshow(framelist[i].reshape(72,100))
        except TypeError:
            #if loading GPU frames, reshaping throws an error, convert to NumPy
            frame = framelist[i].get()
            plt.imshow(frame.reshape(72,100))
            plt.title('Frame {}'.format(i))
            plt.savefig(frameDir+'/frame{}.png'.format(i))

        for agent in models.keys():
            #calculate pixel scores in the frame
            if params.GPU:
                scoreMatrix = makeMapGPU(framelist[i],models[agent],params,i)
                featureMatrix = makeMapNot(framelist[i],models[agent],params,i)

            else:
                scoreMatrix = makeMapCPU(framelist[i],models[agent],params,i)
                featureMatrix = makeMapNotCPU(framelist[i],models[agent],params,i)

            #plot saliency map
            plt.clf()
            saliencyMap = sns.heatmap(scoreMatrix,robust=True,cmap=plt.cm.get_cmap("jet"),xticklabels=False,yticklabels=False)
            plt.title('{} Agent Saliency Map {}'.format(agent,i))
            plt.savefig(params.dir+'/{}/SaliencyMaps/input_map{}.png'.format(agent,i))

            #plot feature map
            plt.clf()
            featureMap = sns.heatmap(featureMatrix,robust=True,cmap=plt.cm.get_cmap("jet"),xticklabels=False,yticklabels=False)
            plt.title('{} Agent Network Feature Map {}'.format(agent,i))
            plt.savefig(params.dir+'/{}/FeatureMaps/feat_map{}.png'.format(agent,i))

        if i>80:break


if __name__== '__main__':
    #retrieve arguments, the frames, and the model weights
    params = make_argparser()
    framelist = loadFrames(params.dir)
    bestModel = loadModel(params.dir)
    worstModel = loadWorstModel(params.dir)
    models = {'Best':bestModel,'Worst':worstModel}

    #create best and worst agent directories
    for agent in models.keys():
        dir = os.path.join(params.dir,agent)
        if not os.path.exists(dir):
            os.makedirs(dir)

        #create a subdirectory to store the saliency maps
        mapDir = os.path.join(dir,'SaliencyMaps')
        if not os.path.exists(mapDir):
            os.makedirs(mapDir)
        
        #create a subdirectory to store the frames
        featDir = os.path.join(dir,'FeatureMaps')
        if not os.path.exists(featDir):
            os.makedirs(featDir)
    
    #create a subdirectory to store the frames
    frameDir = os.path.join(params.dir,'Frames')
    if not os.path.exists(frameDir):
        os.makedirs(frameDir)

    if params.mp:
        import multiprocessing as mp
        from os import getpid
        tgtList = []
        interval = ceil(len(framelist)/params.num_processors)
        length = len(framelist)
        for i in range(params.num_processors):
            if i*interval > length: break
            start = i*interval
            end = (i+1)*interval - 1 if ((i+1)*interval - 1)<length else length-1
            tgtList.append((framelist,start,end,interval,params,models))
        pool = mp.Pool()
        try:
            results = pool.map_async(plotMaps,tgtList)  
        except Exception as e:
            print(e)
        finally:
            pool.close()
            pool.join()
            print('***all processes of {} are done***')
    else:
        #create an input and model saliency map for each loaded frame
        for i in range(0,len(framelist),10):
            print('\n\n*********frame {}**********'.format(i),flush=True)
            plt.close()
            try:
                plt.imshow(framelist[i].reshape(72,100))
            except TypeError:
                #if loading GPU frames, reshaping throws an error, convert to NumPy
                frame = framelist[i].get()
                plt.imshow(frame.reshape(72,100))
                plt.title('Frame {}'.format(i))
                plt.savefig(frameDir+'/frame{}.png'.format(i))

            for agent in models.keys():
                #calculate pixel scores in the frame
            
                if params.GPU:
                    scoreMatrix = makeMapGPU(framelist[i],models[agent],params,i)
                    featureMatrix = makeMapNot(framelist[i],models[agent],params,i)

                else:
                    scoreMatrix = makeMapCPU(framelist[i],models[agent],params,i)
                    featureMatrix = makeMapNotCPU(framelist[i],models[agent],params,i)

                #plot saliency map
                plt.clf()
                saliencyMap = sns.heatmap(scoreMatrix,robust=False,cmap=plt.cm.get_cmap("jet"),xticklabels=False,yticklabels=False)
                plt.title('{} Agent Saliency Map {}'.format(agent,i))
                plt.savefig(params.dir+'/{}/SaliencyMaps/input_map{}.png'.format(agent,i))

                #plot feature map
                plt.clf()
                featureMap = sns.heatmap(featureMatrix,robust=False,cmap=plt.cm.get_cmap("jet"),xticklabels=False,yticklabels=False)
                plt.title('{} Agent Network Feature Map {}'.format(agent,i))
                plt.savefig(params.dir+'/{}/FeatureMaps/feat_map{}.png'.format(agent,i))

            # if i>100:break