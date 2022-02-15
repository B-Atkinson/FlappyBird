# Author: Brian Atkinson

# Created: 1/9/2022

# Description: This code uses FlappyBird as an environment to train RL agents. Agents can learn purely on their own or utilize human reinforcement.
# The environment is provided in the customized pygames learning environment package inside the  parent  directory  of  this  program.  Agents  are
# initialized with a neural network of a single hidden layer and an output layer of one neuron. If human influence is used it slightly modifies the
# probability of the agent moving up prior to sampling from a uniform distribution. Various input parameters and their default values can be  found
# in params.py.

# Attributions: Much of this code was inspired or taken from Brandon Hee's research (as my job was to reproduce a behavior) as well as a blog  post 
# from Andrej Karpathy (https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5).
print('starting FB',flush=True)
import os
import sys
import pickle
import json
import yaml
import csv
import numpy as np
import cupy as cp
from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame
from pygame.constants import K_w
import params


#specified in ple/__init__.py lines 187-194
WIDTH = 100     #downsample by half twice
HEIGHT = 72    #downsample by half twice
GAP = 100
GRID_SIZE = WIDTH * HEIGHT

ACTION_MAP = {
    'flap': K_w,
    'noop': None
}

hparams = params.get_hparams()
#restore hyperparameters from session checkpoint, with tweaked total number of episodes
if hparams.continue_training:
    path = hparams.checkpoint_path
    start = hparams.train_start
    numTotalEpisodes = hparams.num_episodes
    cont = True
    #load the previous hyperparameters
    hparams = pickle.load(open(path+'/hparams.p','rb'))
else:
    cont=False

    
    
rng = np.random.default_rng(hparams.seed)
REWARDDICT = {"positive":hparams.pipe_reward, "loss":hparams.loss_reward}

#### Folders, files, metadata start------------------------------------------------------
#define filepath for saving results
if hparams.human:
    PATH = hparams.output_dir + "/ht-" + str(hparams.num_episodes) + "-S" + str(hparams.seed) + "-loss" + str(hparams.loss_reward) \
        +'-hum'+str(hparams.human_influence)+'-learn'+str(hparams.learning_rate)
else:
    PATH = hparams.output_dir + "/no_ht-" + str(hparams.num_episodes) + "-S" + str(hparams.seed) + "-loss" + \
            str(hparams.loss_reward)+'-learn'+str(hparams.learning_rate)

MODEL_NAME =  PATH + "/pickles/"
ACTIVATIONS = PATH + "/activations/"
STATS = PATH+"/stats.csv"
MOVES = PATH+"/moves.csv"

os.makedirs(os.path.dirname(PATH+'/metadata.txt'), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_NAME), exist_ok=True)
os.makedirs(os.path.dirname(ACTIVATIONS), exist_ok=True)
print('Saving to: ' + PATH,flush=True)

#save metadata for easy viewing 
with open(PATH+'/metadata.txt', 'w') as f:
    json.dump(hparams.__dict__, f, indent=2)
#save metadata object in case of continuing training later
if not hparams.continue_training:
    pickle.dump(hparams, open(PATH+'/hparams.p', 'wb'))

#### Folders, files, metadata end------------------------------------------------------

#### Function Definitions Begin------------------------------------------------------

# Hee's sigmoid function
def sigmoid(value):
    """Activation function used at the output of the neural network."""
    return 1.0 / (1.0 + np.exp(-value)) 
    
# Hee's discounted reward function
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward. """
    discounted_r = cp.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        #reset running sum if encounter a negative number because the last reward is a negative number
        if r[t] != 0:
            running_add = 0

        #discounted reward at this step = (discount_factor * running_sum last step) + reward for this step
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        
    discounted_r -= cp.mean(discounted_r)
    discounted_r /= cp.std(discounted_r)
    return discounted_r

# Karpathy with added normalization and dropout options, from Hee's code
def policy_forward(hparams, screen_input, model):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively)"""
    int_h = cp.dot(model['W1'], screen_input)
    
    if hparams.normalize:
        mean = cp.mean(int_h)
        variance = cp.mean((int_h - mean) ** 2)
        int_h = (int_h - mean) * 1.0 / cp.sqrt(variance + 1e-5)
    
    # ReLU nonlinearity used to get hidden layer state
    int_h[int_h < 0] = 0  
        
    logp = cp.dot(model['W2'], int_h)
    
    #probability of moving the agent up
    p = sigmoid(logp)
    return p, int_h  

# Karpathy's backpropagation functions from Hee's code
def policy_backward(int_harray, grad_array, epx):
    """ backward pass. (int_harray is an array of intermediate hidden states) """
    delta_w2 = cp.dot(int_harray.T, grad_array).ravel()
    delta_h = cp.outer(grad_array, model['W2'])
    delta_h[int_harray <= 0] = 0  # backprop relu
    delta_w1 = cp.dot(delta_h.T, epx)
    return {'W1': delta_w1, 'W2': delta_w2}

# Determine which action to take
def getAction(hparams, FB, observation, model, episode):
    '''Processes the input frame to determine what action to take. '''
    agentMove, hidden_activations = policy_forward(hparams, observation, model)
    state = FB.getGameState()
    deltaX = state['next_pipe_dist_to_player']
    if hparams.human and deltaX>32:        
        humanMove = humanAction(state)
        influence = (hparams.human_influence * (hparams.human_decay ** episode)) if hparams.human_decay else hparams.human_influence
        prob_up = augmentProb(humanMove, agentMove, influence)
    else:
        prob_up = agentMove
    return prob_up, hidden_activations

# # Get the human recommended action choice
# def humanAction(FB):
#     '''The purpose of this function is to take the state of the game and determine a recommended action for the agent to take.
#     The original heuristic was simply to have the agent flap if it was below the gap, and to let gravity pull it down otherwise.
#     This was flawed because the agent takes a somewhat parabolic trajectory, using what resembles projectile motion with a speed 
#     limit. The agent is limited to 10 units of downward speed, and flaps seem to only last for one step, so each time the bird
#     flaps its vertical velocity can effectively be treated as zero. This function models the trajectory of the agent from its 
#     current location out to the right edge of the closest pipe, and determines if the agent would collide with either the ground
#     or the closest pipe if it was to simply not flap until it reaches the pipe. If the trajecotry meets the ground or the lower 
#     pipe, the recommendation is to flap. Otherwise the recommendation is to do nothing. The environment calculates next y position
#     by taking the next velocity and adding it to the current y (i.e. y_next = y_current + vel_next) and the next velocity by sub-
#     tracting 1 from the current, unless the current velocity is negative (upward), in which case it seems to zero out the velocity.
    
#     Input:
#     FB- The FlappyBird game object found in ./ple/games/flappybird/__init__.py
    
#     Output:
#     -The recommended agent action, translated to be immediately usable by the PLE.step() method. 119 if flap, else None. '''
#     #determine the closest pipe in the dictionary 
#     deltaX = 10000
#     for p in FB.pipe_group:
#         dist = (p.x - p.width/2 - 20) - FB.player.pos_x
#         if (dist < deltaX) and (dist>=0):
#             pipe = p
#             deltaX = dist
    
#     #if agent is too far from pipes, allow it to learn on its own
#     if deltaX >= 50:
#         return 'null'

#     #get number of time increments to project over to reach far edge of pipe
#     #the bird moves a constant 4 units closer to the pipes each step
#     if (pipe.width+deltaX) % 4 == 0:
#         steps =  (pipe.width+deltaX) // 4
#     else:
#         steps = 1 + (pipe.width+deltaX) // 4
    
#     #determine the lateral space to the next pipe and where the bird is vertically
#     x = FB.player.pos_x
#     y = FB.player.pos_y
#     v = FB.player.vel
#     if (v<0):
#         #bird flaps do not carry the bird up for more than one time step, change in y is already implemented
#         v = 0
        
#     #calculate number of steps until player velocity reaches 10 downward if no flapping, because velocity limit is 10
#     dif = (10-v -1)     
    
#     x += 4*steps
#     if dif < steps:
#         #if bird reaches downward velocity of 10 before reaching the pipe, the downward velocity becomes fixed at 10
#         # sum of first n numbers = n*(n+1) / 2. 
#         #sum of numbers from m to n, excluding m, where m<n = n(n+1)/2 - m(m+1)/2
#         y += (9*10/2) - (v*(v+1)/2) + 10*(steps-dif)
#         v=10
#     else:
#         #bird will not reach downward velocity of 10 before reaching the pipe
#         y += ((v+steps)*(v+steps+1)/2) - (v*(v+1)/2)
#         v += steps
    
#     #check if agent contacts the ground before or at the left edge of the pipe, tell it to flap
#     if y >= 0.79 * FB.height - FB.player.height:    
#         return ACTION_MAP['flap']
    
#     #projection of agent trajectory through range of x locations where the pipe is, checking if it collides
#     for _ in range(1, 1+pipe.width//4):
#         #check if agent is within the left/right edges of the pipe at any height
#         is_in_pipe = (pipe.x - pipe.width/2 - 20) <= x < (pipe.x + pipe.width/2)
        
#         #check if agent is within the top/bottom edges of the bottom pipe
#         bot_pipe_check = (
#             (FB.player.pos_y +
#              FB.player.height) > pipe.gap_start +
#             FB.pipe_gap) and is_in_pipe        
        
#         #flap if the agent hits the bottom pipe
#         if bot_pipe_check:
#             return ACTION_MAP['flap']            
        
#         #increment positional data and the velocity for next check
#         x += 4
#         v += 1
#         if v>=10:
#             y+=10
#         else:
#             y += v     
#     #if reach this point, agent either collides with the top pipe or passes through, either way it shoud not flap    
#     return ACTION_MAP['noop']

def humanAction(state):
    y = state['player_y']
    
    bottomEdge = state['next_pipe_bottom_y']
    if y>bottomEdge-32:
        #flap if player is in line or below the bottom edge of the gap
        action = ACTION_MAP['flap']
    else:
        #otherwise do nothing
        action = ACTION_MAP['noop']
    return action
    
    

# Get final probability of moving up
def augmentProb(human, agent, influence):
    if human == ACTION_MAP['flap']:
        p_up = agent + influence * agent
    elif human == 'null':
        p_up = agent
    else:
        p_up = agent - influence * agent
    return p_up

def save_csv(data, filename):
    with open(filename, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    csvFile.close()
    
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

def reloadEnvironment(game, FB, rng, hparams, numEpisodes, path):
    '''Replays recorded series of moves through the environment, agent, and random number generator in order to have the 
    randomly-derived values be in the same state that they were when the previous training session ended. The result is 
    to have both individual training sessions have the exact same output as one unbroken session of the same total number 
    of episodes (i.e. 2 sessions of 1,000 episodes produce the same result as a sinlge 2,000 episode session.
    
    Inputs:
    game- a fresh instantiation of the PLE object
    FB- a fresh instantiation of the FlappyBird game object
    rng- a fresh instantiation of the numpy random number generator
    hparams- the parameters object from the previous training session
    numEpisodes- the number of episodes to replay moves through (should be the number in the name of the loaded pickle file)
    
    Outputs:
    game- the PLE environment object as it was at the end of the loaded training session
    FB- the FlappyBird game object as it was at the end of the loaded training session
    rng- the numpy random number generator as it was at the end of the loaded training session
    '''
    with open(path+'/'+'moves.csv',newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        epNum = 1
        for episode in reader:
            game.reset_game()
            FB.resetPipes()   
            FB.init()            
            for move in episode:
                rng.uniform()
                if move == '1':
                    game.act(ACTION_MAP['flap'])
                else:
                    game.act(ACTION_MAP['noop'])
            epNum += 1
            if epNum == numEpisodes:
                break            
    return game, FB, rng
#### End Function Definitions-----------------------------------------------------

#### Environment Setup Begin------------------------------------------------------
if cont:
#load old weights into model
    model = pickle.load(open(path + '/pickles/' + str(start)  + '.p', 'rb'))

else:
    #Intialize the agent weights
    # model - a dictionary whose keys (W1 and W2) have values that represent the connection weights in that layer
    model = {}
    #initialize the weights for the connections between the input pixels and the hidden nodes using a fully-connected method
    model['W1'] = cp.asarray(rng.standard_normal((hparams.hidden,GRID_SIZE)) / np.sqrt(GRID_SIZE))
    #initialize the weights for the connections between the hidden nodes and the single output node using a fully-connected method
    model['W2'] = cp.asarray(rng.standard_normal(hparams.hidden) / np.sqrt(hparams.hidden))
    #save model hyperparameters
    pickle.dump(hparams, open(PATH+'/hparams.p', 'wb'))

#if rendering the game, cannot force the FPS to go faster. 
if not hparams.render:
    #Hamming does not have rendering capability, need a fake output to allow program to run
    #see https://www.py4u.net/discuss/17983
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
#Initialize FB environment   
FLAPPYBIRD = FlappyBird(pipe_gap=GAP, rngSeed=hparams.seed, pipeSeed=hparams.seed+10)
game = PLE(FLAPPYBIRD, display_screen=hparams.render, force_fps=True, rng=hparams.seed, reward_values=REWARDDICT)
game.init()
#### End Environment Setup -------------------------------------------------------


#### Training Begin---------------------------------------------------------------
episode = 1
running_reward = None

#prepare to track episode
#frames- an array that stores each hybrid input frame given to the network
#actions- an array of the actions taken after sampling
#rewards- array of the reward for each step, not cumulative
#activations- an array of hidden layer activation function outputs
episode_actions = []
training_summaries = []
saved_hiddens = []
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

#resume training from previously saved checkpoint
if cont:
    print('priming RNGs',flush=True)
    game, FLAPPYBIRD, rng = reloadEnvironment(game, FLAPPYBIRD, rng, hparams, start, path)
    episode = start+1
    hparams.num_episodes = numTotalEpisodes
    print('done priming\n\n',flush=True)

#regular training loop
while episode <= hparams.num_episodes:
    #reset the pygame object and pipe group list to have the same set of pipes each episode to begin the episode
    game.reset_game()
    FLAPPYBIRD.resetPipes()   
    FLAPPYBIRD.init()
    
    agent_score = 0
    num_pipes = 0
    prev_frame = None       
    frames, actions, rewards, activations, actionTape = [], [], [], [], []
    lastFrame = cp.zeros([GRID_SIZE])
    
    #play episode until environment sets game over variable
    while not game.game_over():
                
        #convert frame to cupy ndarray
        currentFrame = game.getScreenRGB()
        currentFrame = cp.asarray(processScreen(currentFrame))
        
        #create hybrid frame and pass to the network
        if cp.any(lastFrame):
            observation = currentFrame - lastFrame
            lastFrame = currentFrame
        else:
            observation = currentFrame
            lastFrame = currentFrame
        
        # prob_up, hidden_activations = getAction(hparams, FLAPPYBIRD, observation, model, episode)
        prob_up, hidden_activations = getAction(hparams, game, observation, model, episode)
        action = ACTION_MAP['flap'] if rng.uniform() < prob_up else ACTION_MAP['noop']
        reward = game.act(action)
        agent_score += reward
        
        if reward > 0:
            num_pipes += 1
        
        if episode % hparams.hidden_save_rate == 0:
            saved_hiddens.append(hidden_activations)
        
        #record data for this step
        frames.append(observation)
        actions.append(1 if action==K_w else 0) #flaps stored as 1, no-flap stored as 0
        activations.append(hidden_activations)
        rewards.append(reward)
        
        #this tape is used to encourage the action in the future if we see a similar input
        #see http://karpathy.github.io/2016/05/31/rl/#:~:text=looking%20quite%20bleak.-,Supervised%20Learning,-.%20Before%20we%20dive
        actionTape.append(1-prob_up)
    
    
    #episode over, compile all frames' data to prep for backprop   
    episode_actions.append(actions)        
    epx = cp.vstack(frames)             #array of arrays, each subarray is the set of frames for an episode  
    eph = cp.vstack(activations)        #array of arrays, each subarray is the set of hidden layer activations for an episode  
    epr = cp.vstack(rewards)            #array of arrays, each subarray is the set of rewards at each step for an episode  
    epdlogp = cp.vstack(actionTape)     #action encouragement gradient tape of log probability        
    training_summaries.append( (episode, num_pipes) )  #save summary info for this episode to plot later
    
    
    #Save data after episode
    if episode % hparams.hidden_save_rate == 0:
                pickle.dump(saved_hiddens, open(ACTIVATIONS  + str(episode) + '.p', 'wb'))
                saved_hiddens = []
    
    #Do backprop by modulating the gradient with advantage     
    discounted_epr = discount_rewards(epr, hparams.gamma)
    discounted_epr -= cp.mean(discounted_epr)
    discounted_epr /= cp.std(discounted_epr)
    epdlogp *= discounted_epr  
    gradient = policy_backward(eph, epdlogp, epx)
    
    # accumulate grad over batch
    for k in model:
        grad_buffer[k] += gradient[k]  

    # perform rmsprop parameter update every batch_size episodes. Default 10.
    if episode % hparams.batch_size == 0:
            
        w1_before = model['W1']
        for k, v in model.items():
            g = grad_buffer[k]  # gradient
            rmsprop_cache[k] = hparams.decay_rate * rmsprop_cache[k] + (1 - hparams.decay_rate) * g ** 2
            model[k] += hparams.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
            
    #Save and empty the actions and score per episode buffers every X episodes
    if episode % hparams.save_stats == 0:
                #save model weights to pickle file
                pickle.dump(model, open(MODEL_NAME  + str(episode) + '.p', 'wb'))
                #save agent scores and episodes
                with open(STATS, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(training_summaries)
                training_summaries = []
                #save move histories per episode  
                with open(MOVES, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(episode_actions)
                episode_actions = []

    episode += 1
print('training completed',flush=True)
#### End Training-----------------------------------------------------------------