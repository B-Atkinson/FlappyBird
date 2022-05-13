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
from genericpath import exists
import os
import pickle
import json
import csv
import numpy as np
#import cupy as cp
from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame
from pygame.constants import K_w
import params
from matplotlib import pyplot as plt
plt.clf()
magnitudes = {'W1':[],'W2':[]}


hparams = params.get_hparams()
#specified in ple/__init__.py lines 187-194
WIDTH = 100     #downsample by half twice
HEIGHT = 72    #downsample by half twice
GRID_SIZE = WIDTH * HEIGHT
ACTION_MAP = {'flap': K_w,'noop': None}
GAP = 100*hparams.gap_size

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
    PATH = hparams.output_dir + "/ht" + "-S" + str(hparams.seed) + "-Gap" +str(hparams.gap_size)\
        +"-Hyb"+str(hparams.percent_hybrid) +"-FlipH_"+str(hparams.flip_heuristic)+"-Leaky_"+str(hparams.leaky)+\
            "-Init_"+str(hparams.init)+"-Bias"+str(hparams.bias)
else:
    PATH = hparams.output_dir + "/no_ht" + "-S" + str(hparams.seed) + "-Gap" +str(hparams.gap_size)\
        +"-Hyb"+str(hparams.percent_hybrid) +"-FlipH_"+str(hparams.flip_heuristic)+"-Leaky_"+str(hparams.leaky)+\
            "-Init_"+str(hparams.init)+"-Bias"+str(hparams.bias)

try:
    os.makedirs(os.path.dirname(PATH),exist_ok=False)
except FileExistsError:
    #create a unique name for the directory in case of overlapping paths
    print('directory already exists:',PATH)
    from time import time
    PATH += "_"+str(time())[-4:]

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

def policy_forward(hparams, screen_input, model):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively)"""
    int_h = np.dot(model['W1'], screen_input)
    if hparams.bias:
        int_h+=bias
    
    if hparams.normalize:
        mean = np.mean(int_h)
        variance = np.mean((int_h - mean) ** 2)
        int_h = (int_h - mean) * 1.0 / np.sqrt(variance + 1e-5)
    
    if hparams.leaky:
        # # Leaky ReLU 
        int_h[int_h < 0] *= .01
    else:
        # ReLU nonlinearity used to get hidden layer state
        int_h[int_h < 0] = 0      
        
    logp = np.dot(model['W2'], int_h)
    
    #probability of moving the agent up
    p = sigmoid(logp)
    return p, int_h  

# Karpathy's backpropagation functions from Hee's code
def policy_backward(int_harray, grad_array, epx):
    """ backward pass. (int_harray is an array of intermediate hidden states) """
    delta_w2 = np.dot(int_harray.T, grad_array).ravel()
    delta_h = np.outer(grad_array, model['W2'])
    if not hparams.leaky:
        delta_h[int_harray <= 0] = 0  # backprop for regular relu to zero out all negative values
    delta_w1 = np.dot(delta_h.T, epx)
    return {'W1': delta_w1, 'W2': delta_w2}
    
# Hee's discounted reward function
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward. """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        #reset running sum if encounter a negative number because the last reward is a negative number
        if r[t] != 0:
            running_add = 0

        #discounted reward at this step = (discount_factor * running_sum last step) + reward for this step
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r

# Determine which action to take
def getAction(hparams, game, observation, model, episode):
    '''Processes the input frame to determine what action to take. '''
    agentMove, hidden_activations = policy_forward(hparams, observation, model)
    state = game.getGameState()
    if hparams.human and (state['next_pipe_dist_to_player']<=80):        
        # humanMove = humanAction(state)
        if state['player_y'] >(state['next_pipe_bottom_y']-32):
            #flap if player is in line or below the bottom edge of the gap
            if hparams.flip_heuristic: humanMove = ACTION_MAP['noop']
            else: humanMove = ACTION_MAP['flap']
        else:
            #otherwise do nothing
            if hparams.flip_heuristic: humanMove = ACTION_MAP['flap']
            else: humanMove = ACTION_MAP['noop']

        influence = (hparams.human_influence * (hparams.human_decay ** episode)) if hparams.human_decay else hparams.human_influence
        # prob_up = augmentProb(humanMove, agentMove, influence)
        if humanMove == ACTION_MAP['flap']:
            prob_up = min(1,agentMove + influence * agentMove)
        else:
            prob_up = max(0,agentMove - influence * agentMove)
    else:
        prob_up = agentMove
    return prob_up, hidden_activations
    
# Get final probability of moving up
def augmentProb(human, agent, influence):
    if human == ACTION_MAP['flap']:
        p_up = agent + influence * agent
    elif human == 'null':
        p_up = agent
    else:
        p_up = agent - influence * agent
    return p_up
    
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
    if hparams.init == 'Xavier':
        #Xavier initialization
        model['W1'] = rng.standard_normal((hparams.hidden,GRID_SIZE)) / np.sqrt(GRID_SIZE)
        model['W2'] = rng.standard_normal(hparams.hidden) / np.sqrt(hparams.hidden)
    if hparams.init == 'He':
        #He initialization
        model['W1'] = rng.normal(loc=0,size=(hparams.hidden,GRID_SIZE), scale=np.sqrt(2/GRID_SIZE))
        model['W2'] = rng.normal(loc=0,size=hparams.hidden , scale=np.sqrt(2/hparams.hidden))

    #Create a bias vector for the hidden layer
    bias = hparams.bias*np.ones(hparams.hidden)
    #save model hyperparameters
    pickle.dump(hparams, open(PATH+'/hparams.p', 'wb'))

#if rendering the game, cannot force the FPS to go faster. 
if not hparams.render:
    #Hamming does not have rendering capability, need a fake output to allow program to run
    #see https://www.py4u.net/discuss/17983
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

#Initialize FB environment   
FLAPPYBIRD = FlappyBird(pipe_gap=GAP, rngSeed=hparams.seed, pipeSeed=hparams.seed+10)
game = PLE(FLAPPYBIRD, display_screen=False, force_fps=True, rng=hparams.seed, reward_values=REWARDDICT)
game.init()
#### End Environment Setup -------------------------------------------------------


#### Training Begin---------------------------------------------------------------
episode = 1

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

print('commencing training',flush=True)
best_score = -1

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
    lastFrame = np.zeros([GRID_SIZE])
    
    #play episode until environment sets game over variable
    while not game.game_over():
                
        #convert frame to cupy ndarray
        currentFrame = game.getScreenRGB()
        currentFrame = np.asarray(processScreen(currentFrame))
        
        #create hybrid frame and pass to the network
        if np.any(lastFrame):
            observation = currentFrame - hparams.percent_hybrid*lastFrame
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
    
    if num_pipes > best_score:
        pickle.dump(frames,open(PATH+'/bestFrames.p','wb'))
        pickle.dump(model, open(MODEL_NAME  + str(episode) + '.p', 'wb'))
        best_score = num_pipes
    
    
    #episode over, compile all frames' data to prep for backprop   
    episode_actions.append(actions)        
    epx = np.vstack(frames)             #array of arrays, each subarray is the set of frames for an episode  
    eph = np.vstack(activations)        #array of arrays, each subarray is the set of hidden layer activations for an episode  
    epr = np.vstack(rewards)            #array of arrays, each subarray is the set of rewards at each step for an episode  
    epdlogp = np.vstack(actionTape)     #action encouragement gradient tape of log probability        
    training_summaries.append( (episode, num_pipes) )  #save summary info for this episode to plot later
    
    
    #Save hidden layer activations periodically
    if episode % hparams.hidden_save_rate == 0:
                pickle.dump(saved_hiddens, open(ACTIVATIONS  + str(episode) + '.p', 'wb'))
                saved_hiddens = []
    
    #Do backprop by modulating the gradient with advantage     
    discounted_epr = discount_rewards(epr, hparams.gamma)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    epdlogp *= discounted_epr  
    gradient = policy_backward(eph, epdlogp, epx)
    
    # accumulate grad over batch
    for k in model:
        grad_buffer[k] += gradient[k]

    # perform rmsprop parameter update every batch_size episodes
    if episode % hparams.batch_size == 0:
            
        w1_before = model['W1']
        for k, v in model.items():
            gradArray = np.array(grad_buffer[k])
            magnitudes[k].append(np.sqrt(gradArray.dot(gradArray)))
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
    if episode > 400:break

for k,v in magnitudes:
    plt.clf()
    plt.plot(v)
    plt.title('{} Gradient Magnitude'.format(k))
    plt.savefig(PATH+'/{}_gradient.png'.format(k))
print('training completed',flush=True)
#### End Training-----------------------------------------------------------------