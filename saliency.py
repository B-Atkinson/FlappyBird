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
                        help='If true, will utilize Leaky ReLu.')
    parser.add_argument('--dir', type=str,
                        help='The filepath to the test directory to be loaded.')  
    parser.add_argument('--GPU', type=str2bool,default=False,
                        help='If true, will run the code using CuPy.')
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
    digest = os.path.join(dir,'digest.txt')
    if os.path.exists(digest):
        with open(digest) as fd:
            lines = fd.readlines()
        best = lines[1].split(',')[1].split(':')[1]     
        print('best:',best)  
        print('lines:')
        print(lines)
        if not os.path.exists(os.path.join(dir,'pickles/'+best+'.p')):
            #if the exact model is not available, choose the closest saved point
            best = max(floor( int(best) / 100 ) * 100,1)   
    #will throw an exception if the digest file does not exist, fail early
    return str(best)
    

def loadModel(dir):
    gameNumber = findMaxModel(dir)
    file = os.path.join(dir,'pickles/'+gameNumber+'.p')
    model  = pickle.load(open(file,'rb'))
    return model

def loadFrame(file):
    print(file)
    colorFrame = image.imread(file)
    print('\n',np.shape(colorFrame),type(colorFrame))
    img = image.fromarray(colorFrame)
    RGBFrame = img.convert('RGB')
    print('\n',np.shape(RGBFrame),type(RGBFrame),'\n\n')
    plt.imshow(RGBFrame)
    plt.savefig('testvis.png')
    return RGBFrame

def loadFrames(file):
    dir = os.path.join(file,'bestFrames.p')
    print('loading frames from:',dir)
    frameList = pickle.load(open(dir,'rb'))
    print('\n{} frames, each frame shape {}\n\n'.format(len(frameList),cp.shape(frameList[0])))
    # for i in range(len(frameList)):
    #     plt.imshow(np.asarray(frameList[i].get()).reshape(72,100))
    #     plt.savefig('frames/testvis{}.png'.format(i))
    return frameList

def getSaveDir(modelPath):
    return pathlib.PurePath.parents(modelPath)[1]

def processScreen(obs):
    '''Takes as input a 512x288x3 numpy ndarray and downsamples it twice to get a 100x72 output array. Usless background 
       pixels were manually overwritten with 33 in channel 0 to be easier to detect in-situ. Rows 400-512 never change 
       because they're the ground, so they are cropped before downsampling. To reduce the number of parameters of the model,
       only using the 0th channel of the original image.'''
    obs = obs[::2,:400:2,0]
    obs = obs[::2,::2]
    col,row =np.shape(obs)
    for i in range(col):
        for j in range(row):
            #background pixels only have value on channel 0, and the value is 33
            if (obs[i,j]==33):
                obs[i,j] = 0
            elif (obs[i,j]==0):
                pass                
            else:
                obs[i,j] = 1
    return obs.astype(np.float).ravel()

def sigmoid(value):
    """Activation function used at the output of the neural network."""
    return 1.0 / (1.0 + np.exp(-value)) 

def policy_forward(screen_input, model,leaky=False):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively)"""
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

def policy_forward_GPU(screen_input, model,leaky=False):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively)"""
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
    print('frame size:',np.shape(frame))
    input = frame.get()
    if params.GPU:
        blurredImg = cp.asarray(cv.GaussianBlur(input.reshape(72,100),(5,5),cv.BORDER_DEFAULT))
        orig_prob = policy_forward_GPU(frame, model)
        plt.clf()
        plt.imshow(blurredImg.get())
    else:
        blurredImg = cv.GaussianBlur(input.reshape(72,100),(5,5),cv.BORDER_DEFAULT)
        orig_prob = policy_forward(frame, model)
        plt.clf()
        plt.imshow(blurredImg)
    plt.title('Blurred Input Frame')
    plt.savefig('blur.png')
    
    new_prob = []
    for i in range(7200):
        old = frame[i]
        frame[i] = blurredImg[i]
        input = frame.ravel()
        if params.GPU:
            p,_= policy_forward_GPU(input,model,params.leaky)
        else:
            p,_= policy_forward(input,model,params.leaky)
        new_prob.append(p)
        frame[i] = old
    scores = np.array(list(map(lambda i: .5*(orig_prob-i)**2,new_prob))).reshape(72,100)
    print('# pixel scores:',np.shape(scores))
    return scores 



if __name__== '__main__':
    params = make_argparser()
    framelist = loadFrames(params.dir)
    model = loadModel(params.dir)
    # for i,frame in enumerate(framelist):
    #     print(i,':',policy_forward_GPU(frame,model,params.leaky)[0])
    scoreMatrix = makeMap(framelist[10],model,params)
    

    # saliencyMap = cv.applyColorMap(scoreMatrix,cv.COLORMAP_JET)
    # plt.clf()
    # plt.imshow(saliencyMap)
    # plt.title('Saliency Map')
    # plt.savefig(str(getSaveDir(params.model))+'/SaliencyMap.png')