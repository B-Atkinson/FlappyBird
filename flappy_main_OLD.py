"""
Brian Atkinson, Mike Calnan, Nate Haynes

The general idea behind our strategy was to heavily reward the agent whenever it was aligned with the next 
opening in the pipe. This worked, but it took a large number of epochs to get decent results. So we decided 
to amplify that condition, and added additional rewards to get the agent to maintain its position in the
center of the next gap (i.e. if it was closer to one edge than the other, it would get rewarded for having
a velocity that took it away from the edge. This way the agent would not only try to be in the gap, but it 
would essentially target the dead center of the gap and try to maintain that position.

We tried adding extra parameters to punish the agent whenever it was located outside the gap with a velocity
that would take it further from the gap, and reward it if it was outside but has a velocity that would take 
it to the center of the gap. This seemed like it should have worked but we had noticeably worse results when
we tried it.

Attributions:
https://www.geeksforgeeks.org/how-to-get-the-n-largest-values-of-an-array-using-numpy/
"""


import os
import sys
import random
import numpy as np
from IPython import embed
from ple.games.flappybird import FlappyBird
from ple import PLE
from pygame.constants import K_w

WIDTH = 288
HEIGHT = 512
GAP = 100
SEED = 1234

N_PARAMS = 8

#helper references
POSITION = 0
VELOCITY = 1
NEXT_PIPE = 2
NEXT_TOP = 3
NEXT_BOTTOM = 4

# Hyperparameters
MUTATION_RATE = .8
MAGNITUDE = 1
POPULATION = 10
CENTER_ZERO = 0

# Scoring incentive
OPENING = 10
REL_POS = 5

ACTION_MAP = {
    1: K_w,
    0: None
}

FLAPPYBIRD = FlappyBird(width=WIDTH, height=HEIGHT, pipe_gap=GAP)

def normalize(obs):
    x = [
        obs['player_y']/HEIGHT,
        obs['player_vel']/HEIGHT,
        obs['next_pipe_dist_to_player']/WIDTH,
        obs['next_pipe_top_y']/HEIGHT,
        obs['next_pipe_bottom_y']/HEIGHT,
        obs['next_next_pipe_dist_to_player']/WIDTH,
        obs['next_next_pipe_top_y']/HEIGHT,
        obs['next_next_pipe_bottom_y']/HEIGHT
    ]

    return np.array(x)


def agent(x, w):
    '''
    Perceptron agent flaps if x dot w is >= 1

    x is the observed state
    w is the weight vector
    '''
    
    try:
        return 0 if x @ w < 0.5 else 1
    except:
        return 0

def initialize(n_agents=10):
    '''
    Initialize the population
    '''
    
    return np.random.normal(size=(POPULATION,N_PARAMS))
    

def fitness(w, seed=SEED, headless=True):
    '''
    Evaluate the fitness of an agent with the game

    game is a PLE game
    agent is an agent function
    '''
    # disable rendering if headless
    if headless:
        display_screen=False
        force_fps=True
    else:
        display_screen=True
        force_fps=False

    game = PLE(FLAPPYBIRD, display_screen=display_screen, force_fps=force_fps, rng=seed)
    game.init()
    game.reset_game()
    FLAPPYBIRD.rng.seed(seed)

    agent_score = 0
    
    while True:
        if game.game_over():
            break

        x = normalize(game.getGameState())
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        #fitness logic goes here
        # counts distance agent survives for
        agent_score = agent_score + reward
        
        
        ### For Dr Xie ###
       
        '''This subsection would serve as the human heuristic for flappy bird. My understanding when I wrote this a few quarters
        ago was much more rudimentary than now, so it would look different. In essence, what we were trying to do was modify the 
        total score of the agent frame-by-frame using basic logic to aid the agent in playing. This logic would be different 
        common-sense rules such as: 
        -rewarding when the agent is vertically aligned with the next gap
        -if in the gap, reward the agent for having a velocity that moves it toward the exact center of the gap
        -negatively rewarding the agent when not vertically aligned with the next gap
        -if the agent is below the gap, reward the agent for having a velocity that would take it closer to the gap (and vice versa for abover)
        
        We played around with different heuristics and their values, and ultimately found that only using the first bullet led to 
        the best results. The overall heuristic could be said to have two parts: a) keep the bird vertically aligned with the next gap 
        it must pass through, b) try to keep the bird in the exact center of each gap'''
        
        #if bottom of next opening < agent position < top of next opening
        if (x[NEXT_BOTTOM] < x[POSITION] < x[NEXT_TOP]):
            agent_score = agent_score + OPENING
            
            dist_to_top = x[NEXT_TOP] - x[POSITION]
            dist_to_bot = x[POSITION] - x[NEXT_BOTTOM]
            
            #fine tune reward to favor movement toward the exact middle of the opening
            if (dist_to_top < dist_to_bot) and (x[VELOCITY] <= 0):
                agent_score = agent_score + REL_POS
                
            elif (dist_to_bot < dist_to_top) and (x[VELOCITY] > 0):
                agent_score = agent_score + REL_POS
         
                    
    return agent_score


def crossover(w1, w2):
    '''
    Generate an offspring from two agents
    '''  
    # create a mask by choosing 4 random choices from 0's and 1's
    keep1 = np.random.choice([0,1], size=N_PARAMS) 
    
    # take the complement of the previous mask
    keep2 = 1 - keep1
    
    # element wise multiplication plus element
    return w1 * keep1 + w2 * keep2


def mutate(w):
    '''
    Apply random mutations to an agent's genome
    '''
    # TODO: your mutation logic goes here
    
    # controlled by rate (probably of mutating one of the element) and magnititude (how much you modify one of the elements)
    # one COA: add an amount to each element, where that amount is drawn from a normal distribution
    
    # masking off only elements you want to mutate
    # creates a boolean array with a coin flip
    mask = np.random.uniform(size=N_PARAMS) < MUTATION_RATE
    
    # make draws from another distribution which are what we are adding to our agent
    # should come from normal distribution or one centered on zero ie (-1,1)
    # want to move in each direction not just positive or negative with equal probability
    # magnitude you can control with the scale which is std deviation
    # location is the mean
    amount = np.random.normal(size=N_PARAMS, loc=CENTER_ZERO, scale=MAGNITUDE)
    
    # implement mutation
    # mask is masking off amounts the ones we don't want to change and keeping the ones we do want to change
    x = w + mask * amount
    
    return x


def train_agent(n_agents=10, n_epochs=100, headless=True):
    '''
    Train a flappy bird using a genetic algorithm
    '''
    # TODO: genetic algorithm steps below

    # initialization
    population = initialize(n_agents)

    for i in range(n_epochs):
        # evaluate fitness
        fits = [fitness(w, headless=headless) for w in population]
        print(fits)
        
        # 1. Selection
        
        # get an list of indicies sorted in ascending order
        winner_indicies = np.argsort(fits)
        
        # slice and choose the 4 winners
        winners = population[winner_indicies][-4:]
        
        # Randomly choose a top 4 agent and clone it
        clones = [population[random.choice(winner_indicies)] for _ in range(4)]

        # 2. crossover
        # Random children
        children = []
        for _ in range(3):
            parent1 = population[random.choice(winner_indicies)]
            parent2 = population[random.choice(winner_indicies)]
            
            children.append(crossover(parent1, parent2))
            
        # get winner_child
        winner1 = population[winner_indicies][-1]
        winner2 = population[winner_indicies][-2]
        children.append(crossover(winner1, winner2))
        
        
        # 3. mutate children and clones
        for i in range(4):
            children[i] = mutate(children[i])
            clones[i] = mutate(clones[i])
        
         # insertion
        population[0] = winner1
        population[1] = winner2
        
        for i in range(4):
            population[2*i+2] = children[i] # all even
            population[2*i+3] = clones[i]   # all odd
        
        # Must have new population same size as old population
        # 1. keep direct clones of two best agents ( no mutation)
        # 2. random children x 3 
        #       take best five agents and randomly choose two parents three times crossover  (then mutate)
        # 3. winner child x1 (take best two agents from #1 crossover then mutate)
        # 4. clones x4 (clone and mutate randomly selecting from the "best" agents four times)

    # return the best agent found
    fits = [fitness(w, headless=headless) for w in population]
    
    # get an list of indicies sorted in ascending order
    winner_indicies = np.argsort(fits)
    
    # slice and choose the winner
    best_agent = population[winner_indicies][-1]    

    return best_agent


def main(w, seed=SEED):
    '''
    Let an agent play flappy bird
    '''
    game = PLE(FLAPPYBIRD, display_screen=True, force_fps=False, rng=seed)
    game.init()
    game.reset_game()
    FLAPPYBIRD.rng.seed(seed)

    agent_score = 0
    num_frames = 0

    while True:
        if game.game_over():
            break

        x = normalize(game.getGameState())
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        if reward > 0:
            agent_score += 1

        num_frames += 1
        
                   
    print('Frames  :', num_frames)
    print('Score   :', agent_score)   


if __name__ == '__main__':      
    np.random.seed(2345)
    w = train_agent(headless=True)
    main(w)