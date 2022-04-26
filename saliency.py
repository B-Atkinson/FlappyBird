import pathlib
# import pygame
# from pygame.constants import K_w
# from ple.games.flappybird import FlappyBird
# from ple import PLE
# import csv
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.image as image
import cv2 as cv
import pickle
import argparse
import os
from math import floor

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
    dir = os.path.join(dir,'bestFrames.p')
    print('loading frames from:',dir)
    frameList = pickle.load(open(dir,'rb'))
    print('\n{} frames, each frame shape {}'.format(len(frameList),cp.shape(frameList[0])))
    return frameList

# def processScreen(obs):
#     '''Takes as input a 512x288x3 numpy ndarray and downsamples it twice to get a 100x72 output array. Usless background 
#        pixels were manually overwritten with 33 in channel 0 to be easier to detect in-situ. Rows 400-512 never change 
#        because they're the ground, so they are cropped before downsampling. To reduce the number of parameters of the model,
#        only using the 0th channel of the original image.'''
#     obs = obs[::2,:400:2,0]
#     obs = obs[::2,::2]
#     col,row =np.shape(obs)
#     for i in range(col):
#         for j in range(row):
#             #background pixels only have value on channel 0, and the value is 33
#             if (obs[i,j]==33):
#                 obs[i,j] = 0
#             elif (obs[i,j]==0):
#                 pass                
#             else:
#                 obs[i,j] = 1
#     return obs.astype(np.float).ravel()

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

def makeMap(frame,model,params):
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
    input = frame.get()
    if params.GPU:
        blurredImg = cp.asarray(cv.GaussianBlur(input.reshape(72,100),(5,5),cv.BORDER_DEFAULT)).ravel()
        orig_prob,_ = policy_forward_GPU(frame, model,params.leaky)
        orig_prob = orig_prob.get()
        # img = blurredImg.get().reshape(72,100)
        # plt.clf()
        # plt.imshow(img)
    else:
        blurredImg = cv.GaussianBlur(input.reshape(72,100),(5,5),cv.BORDER_DEFAULT).ravel()
        orig_prob,_ = policy_forward(frame, model,params.leaky)
        # plt.clf()
        # plt.imshow(blurredImg.reshape(72,100))
    
    print('\n\n*********frame**********')
    print('original probability:',orig_prob)
    new_prob = []
    for i in range(7200):
        old = frame[i]
        frame[i] = blurredImg[i]
        if params.GPU:
            p,_= policy_forward_GPU(frame,model,params.leaky)
            p = p.get()
        else:
            p,_= policy_forward(frame,model,params.leaky)
        new_prob.append(p)
        frame[i] = old
    
    scores = list(map(lambda i: .5*(orig_prob-i)**2,new_prob))
    print('before normalizing',np.shape(scores))
    print('mean: {:.5f} median: {:.5f} min: {:.5f} max: {:.5f}'.format(np.mean(new_prob),np.median(new_prob),np.min(new_prob),np.max(new_prob)))
    print('mean score: {:.5f} median score: {:.5f} min: {:.5f} max: {:.5f}'.format(np.mean(scores),np.median(scores),np.min(scores),np.max(scores)))

    normalScores = np.array(list(map(lambda j: 0 if j < 10**-6 else j, scores)))
    print('\nafter normalizing')
    print('mean score: {:.5f} median score: {:.5f} min: {:.5f} max: {:.5f}'.format(np.mean(normalScores),np.median(normalScores),np.min(normalScores),np.max(normalScores)))

    return normalScores.reshape(72,100)



if __name__== '__main__':
    #retrieve arguments, the frames, and the model weights
    params = make_argparser()
    framelist = loadFrames(params.dir)
    model = loadModel(params.dir)

    #create a subdirectory to store the saliency maps
    mapDir = os.path.join(params.dir,'SaliencyMaps')
    if not os.path.exists(mapDir):
        os.makedirs(mapDir)

    #create a saliency map for each loaded frame
    for i in range(0,len(framelist),20):
        #calculate pixel scores in the frame
        scoreMatrix = makeMap(framelist[i],model,params)
        # plt.savefig('blur{}.png'.format(i))
        
        plt.cla()
        saliencyMap = cv.applyColorMap(scoreMatrix.astype(np.uint8),cv.COLORMAP_JET)
        
        # #overlay the saliency map on the frame, and save to disk
        # try:
        #     plt.imshow(framelist[i].reshape(72,100))
        # except TypeError:
        #     #if loading GPU frames, reshaping throws an error, convert to NumPy
        #     frame = framelist[i].get()
        #     plt.imshow(frame.reshape(72,100))
        plt.imshow(saliencyMap,alpha=.5)
        plt.title('Saliency Map {}'.format(i))
        plt.savefig(mapDir+'/map{}.png'.format(i))
        
        if i > 60: break