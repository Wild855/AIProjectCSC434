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
LR = 0.0005              # learning rate


class Agent:
    #how far the player can see ahead of itself
    global sight_distance 
    sight_distance = 5
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0            # for randomness
        self.gamma = 0.9            # discount rate
        self.memory = deque(maxlen = MAX_MEMORY)        # holds the actions, pops left once we exceed the max memory capacity
        #TODO - make this better (not hardcoded)
        self.model = Linear_QNet(1, 256, 2)
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
        current_tile_x_pos = 0
        first_tile_current_x_pos = main.spriteList[0].rect.left
        for element in main.spriteList:
            if(element.rect.left <= game.rect.left):
                #DEBUG
                #print("HEY WE'RE HERE")
                #print("current x pos: " + str(current_tile_x_pos) + " new potential x pos: " + str(element.rect.left))
                current_tile_x_pos = max(current_tile_x_pos, element.rect.left)
        #DEBUG
        #print("current tile: " + str(int((current_tile_x_pos - first_tile_current_x_pos) / 32)))
        player_offset_from_tile = game.rect.left - current_tile_x_pos
        current_tile_x_pos = int((current_tile_x_pos - first_tile_current_x_pos) / 32)
        
        min_distance = 3200
        for j in range(13, 18):
            for i in range(current_tile_x_pos, current_tile_x_pos+sight_distance):
                if i < len(main.platformList[j]):
                    val = int(main.platformList[j][i])
                    #print("IS SPIKE? " + str(val == 3))
                    if(val == 3):
                        distance = i-current_tile_x_pos
                        if distance > 0:
                            min_distance = min(distance, min_distance)
        state.append(float(min_distance)/100)
        #state.append(player_offset_from_tile)
        #state.append(game.rect.centery)
        state = np.array(state, dtype=float)
        #print("about to print state:")
        #DEBUG
        #for element in state:
        #    print(str(element))
        #print("done printing state")

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
        #print("size: ", len(mini_sample))

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
        self.epsilon = 160 - self.number_of_games
        final_move = [0,0]

        if random.randint(0, 8000) < self.epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)   # creates multi-dimensional matrix containing floats
            prediction = self.model(state0)                     # model makes a prediction based on the given state
            #print("Model's prediction: " + str(prediction))
            move = torch.argmax(prediction).item()              # returns the indices of the maximum value of all elements in the input tensor
            #print("Model's final move: " + str(move))
            final_move[move] = 1

        return final_move


'''
Training method
'''
def train():
    # DEBUG
    #print("the start of train()")
   
    main.reset()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    current_game_score = 0


    while True:
        
        state_old = agent.get_state(main.player)                   # get the old state

        final_move = agent.get_action(state_old)            # get move

        # DEBUG
        #print("about to call player.update")
        reward, done, score = main.player.update(final_move)    # perform move and get new state
        current_game_score = max(current_game_score, score)
        #print("NEW SCORE " + str(current_game_score))

        # DEBUG
        #print("finished updating")
        state_new = agent.get_state(main.player)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)    # train short memory

        agent.remember(state_old, final_move, reward, state_new, done)

        # We aren't reaching this point because done is not True 05/03/2023 -SW
        if done:
            # DEBUG
            #print("done here?")
            # train long memory and plot result
            main.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            # save the best score
            if current_game_score > record:
                record = current_game_score
                # agent.model.save()

            # Print results
            #print('Game', agent.number_of_games, 'Score', current_game_score, 'Record', record)

            # update scores
            plot_scores.append(current_game_score)
            total_score += current_game_score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            current_game_score = 0


# entry point
if __name__ == '__main__':
    # DEBUG
    #print("training started")
    train()
