import pathlib
import pygame
from pygame.constants import K_w
from ple.games.flappybird import FlappyBird
from ple import PLE
# import csv
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.image as image
import cv2 as cv
import pickle
import argparse

def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments to run analysis for FlappyBird reinforcement learning with human influence.')    
    parser.add_argument('--frame', type=str,
                        help='The filepath to the frame to be loaded.')        
    parser.add_argument('--model', type=str,
                        help='The filepath to the model to be loaded.')  
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

def loadModel(file):
    model  = pickle.load(open(file,'rb'))
    return model

def loadFrame(file):
    print(file)
    colorFrame = image.imread(file)
    print('\n',np.shape(colorFrame),type(colorFrame))
    img = image.fromarray(colorFrame)
    RGBFrame = img.convert('RGB')
    print('\n',np.shape(RGBFrame),type(RGBFrame),'\n\n')
    # frame = colorFrame[:,:,0]
    plt.imshow(RGBFrame)
    plt.savefig('testvis.png')
    return RGBFrame

def loadFrames(file):
    print(file)
    frameList = pickle.load(open(file,'rb'))
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

def makeMap(frame,model):
#   cols,rows  = np.shape(frame)
    print('frame size:',np.shape(frame))
    input = frame.get()
    blurredImg = cp.asarray(cv.GaussianBlur(input.reshape(72,100),(5,5),cv.BORDER_DEFAULT))
    plt.clf()
    plt.imshow(blurredImg.get())
    plt.title('Blurred Input Frame')
    plt.savefig('blur.png')
    orig_prob = policy_forward(frame, model)
    new_prob = []
#   for j in range(cols):
#     for i in range(rows):
#       old = frame[j,i]
#       frame[j,i] = blurredImg[j,i]
#       input = frame.ravel()
#       p,_= policy_forward(input, model)
#       new_prob.append(p)
#       frame[j,i] = old

    for i in range(7200):
        old = frame[i]
        frame[i] = blurredImg[i]
        input = frame.ravel()
        p,_= policy_forward(input, model)
        new_prob.append(p)
        frame[i] = old
    scores = np.array(list(map(lambda i: .5*(orig_prob-i)**2,new_prob))).reshape(72,100)
    # test = np.array(list(map(lambda i: .5*(orig_prob-i)**2,new_prob))).reshape(72,100)

    print('# pixel scores:',np.shape(scores))
    # return (scores.reshape(cols,rows)).astype(np.uint8),test
    return scores #,test



if __name__== '__main__':
    params = make_argparser()
    # frame = loadFrame(params.frame)
    framelist = loadFrames('/home/brian.atkinson/thesis/data/gradient_test/ht-S5-Gap1.4-Hyb1.0-FlipH_False-Leaky_True-Init_Xavier-Bias0_9286/bestFrames.p')
    model = loadModel(params.model)
    for i,frame in enumerate(framelist):
        print(i,':',policy_forward(frame,model,True)[0])
    scoreMatrix = makeMap(framelist[3],model)

    # saliencyMap = cv.applyColorMap(scoreMatrix,cv.COLORMAP_JET)
    # plt.clf()
    # plt.imshow(saliencyMap)
    # plt.title('Saliency Map')
    # plt.savefig(str(getSaveDir(params.model))+'/SaliencyMap.png')