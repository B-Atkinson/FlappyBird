import os
import sys
import random
import numpy as np
#import cupy as cp
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w
import params

# # Set the GPU to use
# cp.cuda.Device(args.gpu).use()

GRID_SIZE = 80 * 80
LEARNING_RATE = 1e-4
GAMMA = 0.99
DECAY_RATE = 0.99 

WIDTH = 288
HEIGHT = 512
GAP = 100
SEED = 1234

#helper references
POSITION = 0
VELOCITY = 1
NEXT_PIPE = 2
NEXT_TOP = 3
NEXT_BOTTOM = 4

ACTION_MAP = {
    1: K_w,
    0: None
}

#Intialize the agent
def initialize(n_agents=10):
    '''
    Initialize the population
    '''
    
    return np.random.normal(size=(POPULATION,N_PARAMS))

#Initialize FB environment
FLAPPYBIRD = FlappyBird(width=WIDTH, height=HEIGHT, pipe_gap=GAP)

#Do training loop
    #Do an episode
    
    #Do backprop
    
    #Save score
    
    #Record network weights every X episodes
    


