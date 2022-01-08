import os
import sys
import random
import pickle
import yaml
import numpy as np
#import cupy as cp
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w
import params

# # Set the GPU to use
# cp.cuda.Device(args.gpu).use()

#specified in ple/__init__.py lines 187-194
WIDTH = 288
HEIGHT = 512
GAP = 100
GRID_SIZE = WIDTH * HEIGHT

ACTION_MAP = {
    'flap': K_w,
    'noop': None
}

hparams = params.get_hparams()


#### Function Definitions Begin------------------------------------------------------

# Hee's sigmoid function
def sigmoid(value):
    """Activation function used at the output of the neural network."""
    return 1.0 / (1.0 + np.exp(-value)) 
    
# Hee's discounted reward function
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward. """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0

        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r

# Karpathy with added normalization and dropout options, from Hee's code
def policy_forward(screen_input, model, hparams):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively)"""
    int_h = np.dot(model['W1'], screen_input)
    
    if hparams.normalize:
        mean = np.mean(int_h)
        variance =np.mean((int_h - mean) ** 2)
        int_h = (int_h - mean) * 1.0 / np.sqrt(variance + 1e-5)
    
    # ReLU nonlinearity used to get hidden layer state
    int_h[int_h < 0] = 0  
    
    # dropout functionality to randomly remove 
    if hparams.dropout != 0:
        mask = np.random.binomial(1, 1-hparams.dropout) * (1.0/(1-hparams.dropout))
        int_h = int_h * mask
        
    logp = np.dot(model['W2'], int_h)
    
    #probability of moving the agent up
    p = sigmoid(logp)
    return p, int_h  

# Karpathy's backpropagation functions from Hee's code
def policy_backward(int_harray, grad_array, epx):
    """ backward pass. (int_harray is an array of intermediate hidden states) """
    delta_w2 = np.dot(int_harray.T, grad_array).ravel()
    delta_h = np.outer(grad_array, model['W2'])
    delta_h[int_harray <= 0] = 0  # backprop relu
    delta_w1 = np.dot(delta_h.T, epx)
    return {'W1': delta_w1, 'W2': delta_w2}

# Determine which action to take
def getAction(hparams, observation, model):
    ...

# Get the human recommended action choice
def humanAction(game):
    #get dictionary of state values describing the game, see FlappyBird/__init__.py lines 313-360
    state = game.getGameState()
    
    #keep in mind the y axis is flipped, y origin is the ceiling and +y points down, -y goes up
    position = state['player_y']
    velocity = state['player_vel']
    topNextGap = state['next_pipe_top_y']
    bottomNextGap = state['next_pipe_bottom_y']
    
    if position < bottomNextGap:
        #agent is in the gap  or above it, do nothing to fall into the gap
        action = ACTION_MAP['noop']
    # elif bottomNextGap <= position:
    else:
        #agent is in line with the bottom edge or below it, need to flap
        action = ACTION_MAP['flap']
    return action
    

# Get final probability of moving up
def augmentProb():
    ...


#### Environment Setup Begin------------------------------------------------------

#Intialize the agent weights
# model - a dictionary whose keys (W1 and W2) have values that represent the connection weights
#         in that layer
model = {}

#initialize the weights for the connections between the input pixels and the hidden nodes
#using a fully-connected method
model['W1'] = np.random.randn(hparams.hidden,GRID_SIZE) / np.sqrt(GRID_SIZE)
    
#initialize the weights for the connections between the hidden nodes and the single output node
#using a fully-connected method
model['W2'] = np.random.randn(hparams.hidden) / np.sqrt(hparams.hidden)


#Initialize FB environment
#if rendering the game, cannot force the FPS to go faster. 
FLAPPYBIRD = FlappyBird(width=WIDTH, height=HEIGHT, pipe_gap=GAP, rngSeed=hparams.seed)
game = PLE(FLAPPYBIRD, display_screen=hparams.render, force_fps=not hparams.render, rng=hparams.seed)
game.init()
game.reset_game()


#### Training Begin------------------------------------------------------------

episode = 1
prev_frame = None       #will use to compute the hybrid frame
running_reward = None
reward_sum = 0          #store cumulative reward for the episode

#prepare to track episode
#frames- an array that stores each hybrid input frame given to the network
#actions- an array of the actions taken after sampling
#rewards- array of the reward for each step, not cumulative
#activations- an array of hidden layer activation function outputs
frames, actions, rewards, activations = [], [], [], []



#Do training loop
while episode <= hparams.num_episodes:
    agent_score = 0
    
    #Do an episode
    while not game.game_over():
        observation = game.getScreenGrayscale()
        #preprocess?
        
        # action = call function to decide an action
        action = getAction(hparams, observation, model)
        reward = game.act(action)
    
    game.reset_game()
    episode += 1
    
    #Save score

    #Do backprop    
    
    #Record network activations every X episodes
    


