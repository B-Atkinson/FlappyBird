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
from math import floor
from copy import deepcopy


def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run analysis for FlappyBird reinforcement learning with human influence.')    
    parser.add_argument('--leaky', type=str2bool,default=False,
                        help='If true, will utilize Leaky ReLu activation function.')
    parser.add_argument('--dir', type=str,
                        help='The filepath to the test directory to be loaded.')  
    parser.add_argument('--GPU', type=str2bool,default=False,
                        help='If true, will run the code using CuPy and NumPy. This is required to analyze results from GPU-based tests.')
    parser.add_argument('--saveTo', type=str,
                        help='The filepath to the directory where saliency maps should be saved.')  
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
        if not os.path.exists(bestPath):
            #if the exact model is not available, choose the next highest saved point without going over
            print('\n***True best model unavailable, choosing closest option.***\n\n')
            best = max(floor( int(best) / 100 ) * 100,1)   
    #will throw an exception if the digest file does not exist, fail early
    return str(best)
    
def loadModel(dir):
    '''Takes as input the filepath to the test directory, and returns the best model available based on the search completed by findMaxModel. The
    returned object is a Numpy ndarray if the test was run purely on CPUs, and is a CuPy array if the test was done on GPUs.'''
    gameNumber = findMaxModel(dir)
    file = os.path.join(dir,'pickles/'+gameNumber+'.p')
    model  = pickle.load(open(file,'rb'))
    return model

def loadFrames(dir):
    '''Takes as input the filepath to the test directory, and returns a homogenous python list containing the frames from the best performing episode of that test.
    Each item in the returned list is either a Numpy array (if the test was done purely on CPUs) or a CuPy array (if the test was done on GPUs).'''
    try:
        dir = os.path.join(dir,'bestFrames.p')
        print('loading frames from:',dir)
        frameList = pickle.load(open(dir,'rb'))
        print('\n{} frames, each frame shape {}'.format(len(frameList),cp.shape(frameList[0])))
    except FileNotFoundError:
        print('no best frames in test, using another batch')
        frameList = pickle.load(open('/home/brian.atkinson/thesis/data/gradient_test/ht-S5-Gap1.4-Hyb1.0-FlipH_False-Leaky_True-Init_Xavier-Bias0_6785/bestFrames.p','rb'))
        print('\n{} frames, each frame shape {}'.format(len(frameList),cp.shape(frameList[0])))
    return frameList

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

    # #uncomment this section if you want to verify that the frame isn't being clobbered during read/writes
    # checksum = np.sum(origFrame)

    blurredImg = cv.GaussianBlur(input,(5,5),cv.BORDER_DEFAULT).ravel()
    orig_prob,_ = policy_forward(frame, model,params.leaky)
    
    if id(blurredImg) == id(frame):
        print('\nThere is an issue with blurring\n')
    
    print('\n\n*********frame {}**********'.format(frameNum))
    print('probability data:')
    print('original probability:',orig_prob)
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

        # #uncomment this section if you want to verify that the frame isn't being clobbered during read/writes
        # newChecksum = np.sum(frame)
        # if newChecksum != checksum:
        # # if not (frame==origFrame).all():
        #     print('\nstep {} difference: {} checksum: {}   copied frame: {}\n'.format(i,checksum-newChecksum,checksum,newChecksum))
        #     raise Exception('\n***copied frame is not equal to original after calculating scores for frame {}***'.format(frameNum))
        # checksum = newChecksum
    
    #apply scoring function to the perturbed frame
    # scores = list(map(lambda i: .5*(orig_prob-i)**2,new_prob))
    scores = list(map(lambda i: abs(orig_prob-i),new_prob))

    #generate some statistical data for analysis
    print('mean: {:.5f} median: {:.5f} min: {:.5f} max: {:.5f}\n'.format(np.mean(new_prob),np.median(new_prob),np.min(new_prob),np.max(new_prob)))
    print('before normalizing the scores',np.shape(scores))
    print('mean: {:.5f} median: {:.5f} min: {:.5f} max: {:.5f}'.format(np.mean(scores),np.median(scores),np.min(scores),np.max(scores)))
    pers = np.percentile(scores,[25,50,75])
    print('1Q: {:.6f}  2Q: {:.6f}  3Q: {:.6f}'.format(pers[0],pers[1],pers[2]))

    with open('../score_files/scores_{}.txt'.format(frameNum),'w') as file:
        file.write('{}\n'.format(str(params.dir)))
        file.write('original prob: {}\n'.format(orig_prob))
        for i in range(len(scores)):
            file.write('pixel prob: {}   score: {}\n'.format(new_prob[i],scores[i]))

    return np.array(scores).reshape(72,100),min,max


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

    # #uncomment this section if you want to verify that the frame isn't being clobbered during read/writes
    # checksum = cp.sum(origFrame)

    blurredImg = cp.asarray(cv.GaussianBlur(input.reshape(72,100),(5,5),cv.BORDER_DEFAULT)).ravel()
    orig_prob,_ = policy_forward_GPU(frame, model,params.leaky)
    orig_prob = orig_prob.get()
    
    # print('\n\n*********frame {}**********'.format(frameNum))
    print('probability data:')
    print('original probability:',orig_prob)
    new_prob = []
    old = cp.array([0])
    for i in range(7200):
        #write the blurred pixel value to the target pixel, save the old value
        old[0] = cp.copy(frame[i])
        frame[i] = cp.copy(blurredImg[i])
        # newCheck = cp.sum(frame)
        # if newCheck!=checksum:
        #     print('new: {} old: {}'.format(newCheck,checksum))

        #get the perturbed probability
        p,_= policy_forward_GPU(frame,model,params.leaky)
        p = p.get()
        #save the result and write the original target pixel value back to the frame 
        new_prob.append(p)
        frame[i] = cp.copy(old[0])

        # #uncomment this section if you want to verify that the frame isn't being clobbered during read/writes
        # newChecksum = cp.sum(frame)
        # if newChecksum != checksum:
        # # if not (frame==origFrame).all():
        #     print('\nstep {} difference: {} checksum: {}   copied frame: {}\n'.format(i,checksum-newChecksum,checksum,newChecksum))
        #     raise Exception('\n***copied frame is not equal to original after calculating scores for frame {}***'.format(frameNum))
        # checksum = newChecksum
    
    #apply scoring function to the perturbed frame
    # scores = list(map(lambda i: .5*(orig_prob-i)**2,new_prob))
    scores = list(map(lambda i: abs(orig_prob-i),new_prob))

    #generate some statistical data for analysis
    print('mean: {:.5f} median: {:.5f} min: {:.5f} max: {:.5f}\n'.format(np.mean(new_prob),np.median(new_prob),np.min(new_prob),np.max(new_prob)))
    print('before normalizing the scores',np.shape(scores))
    print('mean: {:.5f} median: {:.5f} min: {:.5f} max: {:.5f}'.format(np.mean(scores),np.median(scores),np.min(scores),np.max(scores)))
    pers = np.percentile(scores,[25,50,75])
    print('1Q: {:.6f}  2Q: {:.6f}  3Q: {:.6f}'.format(pers[0],pers[1],pers[2]))

    with open('../score_files/scores_{}.txt'.format(frameNum),'w') as file:
        file.write('{}\n'.format(str(params.dir)))
        file.write('original prob: {}\n'.format(orig_prob))
        for i in range(len(scores)):
            file.write('pixel prob: {}   score: {}\n'.format(new_prob[i],scores[i]))

    return np.array(scores).reshape(72,100),min,max


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
    
    #create a blurred image to pull individual pixel values from, which saves computation
    #both branches accomplish the same task, getting the probability
    frame = cp.copy(origFrame)

    # #uncomment this section if you want to verify that the frame isn't being clobbered during read/writes
    # checksum = cp.sum(origFrame)

    orig_prob,_ = policy_forward_GPU(frame, model,params.leaky)
    orig_prob = orig_prob.get()
    
    # print('\n\n*********frame {}**********'.format(frameNum))
    print('probability data:')
    print('original probability:',orig_prob)
    new_prob = []
    old = np.array([0])
    for i in range(7200):
        #negate pixel value
        old = np.copy(frame[i])
        frame[i] = int(not(frame[i]))
        # newCheck = cp.sum(frame)
        # if newCheck!=checksum:
        #     print('new: {} old: {}'.format(newCheck,checksum))

        #get the perturbed probability
        p,_= policy_forward_GPU(frame,model,params.leaky)
        p = p.get()
        #save the result and write the original target pixel value back to the frame 
        new_prob.append(p)
        frame[i] = cp.copy(old)

        # #uncomment this section if you want to verify that the frame isn't being clobbered during read/writes
        # newChecksum = cp.sum(frame)
        # if newChecksum != checksum:
        # # if not (frame==origFrame).all():
        #     print('\nstep {} difference: {} checksum: {}   copied frame: {}\n'.format(i,checksum-newChecksum,checksum,newChecksum))
        #     raise Exception('\n***copied frame is not equal to original after calculating scores for frame {}***'.format(frameNum))
        # checksum = newChecksum
    
    #apply scoring function to the perturbed frame
    # scores = list(map(lambda i: .5*(orig_prob-i)**2,new_prob))
    scores = list(map(lambda i: abs(orig_prob-i),new_prob))

    #generate some statistical data for analysis
    print('mean: {:.5f} median: {:.5f} min: {:.5f} max: {:.5f}\n'.format(np.mean(new_prob),np.median(new_prob),np.min(new_prob),np.max(new_prob)))
    print('before normalizing the scores',np.shape(scores))
    print('mean: {:.5f} median: {:.5f} min: {:.5f} max: {:.5f}'.format(np.mean(scores),np.median(scores),np.min(scores),np.max(scores)))
    pers = np.percentile(scores,[25,50,75])
    print('1Q: {:.6f}  2Q: {:.6f}  3Q: {:.6f}'.format(pers[0],pers[1],pers[2]))

    with open('../score_files/scores_{}.txt'.format(frameNum),'w') as file:
        file.write('{}\n'.format(str(params.dir)))
        file.write('original prob: {}\n'.format(orig_prob))
        for i in range(len(scores)):
            file.write('pixel prob: {}   score: {}\n'.format(new_prob[i],scores[i]))

    return np.array(scores).reshape(72,100),min,max


if __name__== '__main__':
    #retrieve arguments, the frames, and the model weights
    params = make_argparser()
    framelist = loadFrames(params.dir)
    model = loadModel(params.dir)

    #create a subdirectory to store the saliency maps
    mapDir = os.path.join(params.dir,'SaliencyMaps')
    if not os.path.exists(mapDir):
        os.makedirs(mapDir)

    #create a subdirectory to store the frames
    frameDir = os.path.join(params.dir,'Frames')
    if not os.path.exists(frameDir):
        os.makedirs(frameDir)

    #create a saliency map for each loaded frame
    for i in range(0,len(framelist)):
        print('\n\n*********frame {}**********'.format(i),flush=True)
        # print('checksum:',cp.sum(framelist[i]))
        plt.close()
        # #need to overlay the saliency map on the frame, and save to disk
        try:
            plt.imshow(framelist[i].reshape(72,100))
        except TypeError:
            #if loading GPU frames, reshaping throws an error, convert to NumPy
            frame = framelist[i].get()
            plt.imshow(frame.reshape(72,100))
        plt.title('Frame {}'.format(i))
        plt.savefig(frameDir+'/frame{}.png'.format(i))

        #calculate pixel scores in the frame
        if params.GPU:
            scoreMatrix,min,max = makeMapNot(framelist[i],model,params,i)
        else:
            scoreMatrix,min,max = makeMapCPU(framelist[i],model,params,i)
        # if i > 80: break

        #plot saliency map
        plt.clf()
        saliencyMap = sns.heatmap(scoreMatrix,robust=True,cmap=plt.cm.get_cmap("jet"),xticklabels=False,yticklabels=False)
        plt.title('Saliency Map {}'.format(i))
        plt.savefig(mapDir+'/not_map{}.png'.format(i))
        

        
        