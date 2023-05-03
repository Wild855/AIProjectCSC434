"""
This class creates a learning-based agent -- which is used to sense the environment
through sensors. It then uses an internal mechanism of user feedback, and then make changes according to the performance
of the action we are taking, which, when aligned with the learning goals, are transformed into the actions which are to be taken

"""

import math
import torch
import random
import numpy as np
import main
import pygame
from collections import deque
#from game import SnakeGameAI, Direction, Point  <-- Will need to incorporate the Pydash game class (main.py) to replace this
from model import Linear_QNet, QTrainer
from main import Player, reset


from helper import plot

# Predefined variables
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001              # learning rate


class Agent:
    #how far the player can see ahead of itself
    global sight_distance 
    sight_distance = 20
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0            # for randomness
        self.gamma = 0.9            # discount rate
        self.memory = deque(maxlen = MAX_MEMORY)        # holds the actions, pops left once we exceed the max memory capacity
        #TODO - make this better (not hardcoded)
        self.model = Linear_QNet((18*sight_distance*3)+2, 256, 2)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    '''
    Gets the current state of the game
    
    Parameters
    ------------
    game        instance of our game
    '''
    def get_state(self, game):
        # Jump state
        state = []
        #convert pixels into tile position in array
        x_pos = math.floor(game.rect.left / 32)
        

        for row in main.platformList:
            for i in range(x_pos, x_pos+sight_distance):
                if i < len(row):
                    state.append(row[i] == "Spike")
                    state.append(row[i] == "Orb")
                    state.append(row[i] == "0")

        state.append(game.rect.centerx)
        state.append(game.rect.centery)
        state = np.array(state, dtype=int)

        return state
        

        # Danger up and right

    '''
    Associates the old state with the final move, reward, new state, and done, and stores it in the memories
    
    Parameters
    ------------
    state           state we are currently in
    action          action for the model to take
    reward          the reward we get for taking that action
    next_state      the next state 
    done            indicates whether we got a game over
    '''
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, state, done))      # popleft if MAX_MEMORY is reached


    '''
    Trains the model to improve over time
    '''
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)       # chooses 1000 (BATCH_SIZE) unique random samples from memory, and returns a list of tuples
        else:
            mini_sample = self.memory

        # DEBUG
        print("size: ", len(mini_sample))

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    '''
    Trains the model to optimize one step
    
    Parameters
    ------------
    state           state we are currently in
    action          action for the model to take
    reward          the reward we get for taking that action
    next_state      the next state 
    done            indicates if we got a game over
    '''
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    '''
    Random moves: tradeoff exploration / exploitation
    '''
    def get_action(self, state):
        self.epsilon = 80 - self.number_of_games
        final_move = [0,0]

        # Get a random integer between 0 and 200
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)   # creates multi-dimensional matrix containing floats
            prediction = self.model(state0)                     # model makes a prediction based on the given state
            move = torch.argmax(prediction).item()              # returns the indices of the maximum value of all elements in the input tensor
            final_move[move] = 1

        return final_move


'''
Training method
'''
def train():
    # DEBUG
    print("initializing player")
    # sets the frame rate of the program
    clock = pygame.time.Clock()
    main.reset()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()


    while True:
        clock.tick(60)
        state_old = agent.get_state(main.player)                   # get the old state

        final_move = agent.get_action(state_old)            # get move

        # DEBUG
        print("about to call player.update")
        reward, done, score = main.player.update(final_move)    # perform move and get new state

        # DEBUG
        print("finished updating")
        state_new = agent.get_state(main.player)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)    # train short memory

        agent.remember(state_old, final_move, reward, state_new, done)

        # We aren't reaching this point because done is not True 05/03/2023 -SW
        if done:
            # DEBUG
            print("done here?")
            # train long memory and plot result
            main.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            # save the best score
            if score > record:
                record = score
                # agent.model.save()

            # Print results
            print('Game', agent.number_of_games, 'Score', score, 'Record', record)

            # update scores
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


# entry point
if __name__ == '__main__':
    # DEBUG
    print("training started")
    train()
