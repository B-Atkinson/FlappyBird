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
# import cupy as cp
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w
import params

# # Set the GPU to use
# cp.cuda.Device(hparams.gpu).use()

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
rng = np.random.default_rng(hparams.seed)

#### Folders, files, metadata start------------------------------------------------------
PATH = "ht-" if hparams.human else "no_ht"
PATH = PATH + "-" + str(hparams.num_episodes) + "-S" + str(hparams.seed) + "-H" + str(hparams.hidden)
MODEL_NAME =  PATH + "/pickles/"
ACTIVATIONS = PATH + "/activations/"
STATS = "/stats.csv"
MOVES = "/moves.csv"

os.makedirs(os.path.dirname(PATH+'/metadata.txt'), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_NAME), exist_ok=True)
os.makedirs(os.path.dirname(ACTIVATIONS), exist_ok=True)
print('Saving to: ' + PATH)

with open(PATH+'/metadata.txt', 'w') as f:
    json.dump(hparams.__dict__, f, indent=2)
#### Folders, files, metadata end------------------------------------------------------

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
def policy_forward(hparams, screen_input, model):
    """Uses screen_input to find the intermediate hidden state values along
    with the probability of taking action 2 (int_h and p respectively)"""
    int_h = np.dot(model['W1'], screen_input)
    
    if hparams.normalize:
        mean = np.mean(int_h)
        variance = np.mean((int_h - mean) ** 2)
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
def getAction(hparams, game, observation, model, episode):
    '''Processes the input frame to determine what action to take. '''
    agentMove, hidden_activations = policy_forward(hparams, observation, model)
    if hparams.human:        
        humanMove = humanAction(game)
        influence = (hparams.human_influence * (hparams.human_decay ** episode)) if hparams.human_decay else hparams.human_influence
        prob_up = augmentProb(humanMove, agentMove, influence)
    else:
        prob_up = agentMove
    return prob_up, hidden_activations

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
def augmentProb(human, agent, influence):
    if human == ACTION_MAP['flap']:
        p_up = agent + influence * agent
    else:
        p_up = agent - influence * agent
    return p_up

def save_csv(data, filename):
    with open(filename, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    csvFile.close()


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
if not hparams.render:
    #Hamming does not have rendering capability, need a fake output to allow program to run
    #see https://www.py4u.net/discuss/17983
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
FLAPPYBIRD = FlappyBird(width=WIDTH, height=HEIGHT, pipe_gap=GAP, rngSeed=hparams.seed)
game = PLE(FLAPPYBIRD, display_screen=hparams.render, force_fps=not hparams.render, rng=hparams.seed)
game.init()



#### Training Begin------------------------------------------------------------

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

print('starting training',flush=True)

#Do training loop
while episode <= hparams.num_episodes:
    game.reset_game()
    if episode % 10 == 0:
        game.display_screen = True
        game.force_fps = False
    else:
        game.display_screen = False
        game.force_fps = True
    agent_score = 0
    prev_frame = None       #will use to compute the hybrid frame
    frames, actions, rewards, activations, actionTape = [], [], [], [], []
    
    print('episode: {}'.format(episode))
    
    #Do an episode
    while not game.game_over():
        
        observation = game.getScreenGrayscale()
        observation = observation.astype(np.float).ravel()
        
        
        #preprocess to eliminate background values?
        
        # action = call function to decide an action
        prob_up, hidden_activations = getAction(hparams, game, observation, model, episode)
        action = ACTION_MAP['flap'] if rng.uniform() > prob_up else ACTION_MAP['noop']
        reward = game.act(action)
        agent_score += reward
        
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
    epx = np.vstack(frames)             #array of arrays, each subarray is the set of frames for an episode  
    eph = np.vstack(activations)        #array of arrays, each subarray is the set of hidden layer activations for an episode  
    epr = np.vstack(rewards)            #array of arrays, each subarray is the set of rewards at each step for an episode  
    epdlogp = np.vstack(actionTape)     #action encouragement gradient tape of log probability        
    training_summaries.append( (episode, agent_score) )  #save summary info for this episode to plot later
    
    
    #Save data after episode
    if episode % hparams.hidden_save_rate == 0:
                pickle.dump(saved_hiddens, open(ACTIVATIONS  + str(episode) + '.p', 'wb'))
                saved_hiddens = []
    
    #Do backprop    
    discounted_epr = discount_rewards(epr, hparams.gamma)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    epdlogp *= discounted_epr  # modulate the gradient with advantage 
    gradient = policy_backward(eph, epdlogp, epx)
    
    for k in model:
        grad_buffer[k] += gradient[k]  # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes. Default 10.
    if episode % hparams.batch_size == 0:
            
        w1_before = model['W1']
        for k, v in model.items():
            g = grad_buffer[k]  # gradient
            rmsprop_cache[k] = hparams.decay_rate * rmsprop_cache[k] + (1 - hparams.decay_rate) * g ** 2
            model[k] += hparams.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
            
    # #Record network the actions and score per episode every X episodes
    # if episode % hparams.save_stats == 0:
    #             pickle.dump(model, open(MODEL_NAME  + str(episode) + '.p', 'wb'))
    #             # save_csv(training_summaries, STATS); training_summaries = []
    #             with open(STATS, 'a', newline='') as file:
    #                 writer = csv.writer(file)
    #                 writer.writerows(training_summaries)
                    
    #             # save_csv(episode_actions, MOVES); episode_actions = []
    #             with open(MOVES, 'a', newline='') as file:
    #                 writer = csv.writer(file)
    #                 writer.writerows(episode_actions)
    print('episode {0} score: {1}'.format(episode, agent_score),flush=True)
    episode += 1
