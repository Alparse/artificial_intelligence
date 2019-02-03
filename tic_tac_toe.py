"""
Classic tic tac toe system implemented by Serhat Alpar.
Copied from Xxxxxxxxxxxxxxxxx
permalink: Xxxxxxxxxxxxxx
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class TicTacToeEnv():
    """
    Description:
        A game of classical tic tac toe is played until either a draw (all 9 spaces filled without a win condition
        being met) or a win (a player claims three horizontally or vertically connected cells in the game obs_space:

        Game obs_space is represented by a 2 dimensional array of shape 3,3


    Observation Space:
        Type: Box(4)
        Num	Observation                 Values
        0	[0,0]position (P1)          0,1
        1	[0,1]position (P1)          0,1
        2	[0,2]position (P1)          0,1
        3	[1,0]position (P1)          0,1
        4   [1,1]position (P1)          0,1
        5   [1,2]position (P1)          0,1
        6   [2,0]position (P1)          0,1
        7   [2,1]position (P1)          0,1
        8   [2,2]position (P1)          0,1

        9	[0,0]position (P2)          0,1
        10  [0,1]position (P2)          0,1
        11	[0,2]position (P2)          0,1
        12	[1,0]position (P2)          0,1
        13  [1,1]position (P2)          0,1
        14  [1,2]position (P2)          0,1
        15  [2,0]position (P2)          0,1
        16  [2,1]position (P2)          0,1
        17  [2,2]position (P2)          0,1

    Actions:
        Type: Discrete(9)
        Num	Action(Claim)               Values
        0	[0,0]position (P1)          0,1
        1	[0,1]position (P1)          0,1
        2	[0,2]position (P1)          0,1
        3	[1,0]position (P1)          0,1
        4   [1,1]position (P1)          0,1
        5   [1,2]position (P1)          0,1
        6   [2,0]position (P1)          0,1
        7   [2,1]position (P1)          0,1
        8   [2,2]position (P1)          0,1

        9	[0,0]position (P2)          0,1
        10  [0,1]position (P2)          0,1
        11	[0,2]position (P2)          0,1
        12	[1,0]position (P2)          0,1
        13  [1,1]position (P2)          0,1
        14  [1,2]position (P2)          0,1
        15  [2,0]position (P2)          0,1
        16  [2,1]position (P2)          0,1
        17  [2,2]position (P2)          0,1



    Reward
        1 for win
        .5 for draw
        .1 for move
        0 for loss

    Starting State:
        All observations are set to 0

     Episode Termination
        Win/Loss/Draw Condition Met

    """

    def __init__(self):
        self.obs_space = np.zeros([1, 18])
        self.allowed_action_space = np.arange(18)
        self.allowed_total_action_space=np.zeros((1,9))

        self.state = None
        self.game_status = 0
        # game status 0 = playing
        # game status 1 = player 1 win
        # game status 2 = player 2 win
        # game status 3 = draw
        self.turn = 1
        self.yrl_ = np.zeros([1, 9])
        self.reward1 = 0.0
        self.total_rewards1 = 0
        self.reward2 = 0.0
        self.total_rewards2 = 0
        self.reward_dictionary1=[]

    def check_if_move_legal(self, player, move):
        if player == 1:
            if move not in self.allowed_action_space:
                return False
            else:
                return True
        if player == 2:
            # if move not in self.allowed_action_space_player2:
            if move not in self.allowed_action_space:
                return False
            else:
                return True

    def check_if_game_won(self, player):
        if player == 1:
            os = 0  # offset
        else:
            os = 9  # offset
        if 1 == self.obs_space[0, 0 + os] == self.obs_space[0, 1 + os] == self.obs_space[0, 2 + os] \
                or 1 == self.obs_space[0, 3 + os] == self.obs_space[0, 4 + os] == self.obs_space[0, 5 + os] \
                or 1 == self.obs_space[0, 6 + os] == self.obs_space[0, 7 + os] == self.obs_space[0, 8 + os] \
                or 1 == self.obs_space[0, 0 + os] == self.obs_space[0, 3 + os] == self.obs_space[0, 6 + os] \
                or 1 == self.obs_space[0, 1 + os] == self.obs_space[0, 4 + os] == self.obs_space[0, 7 + os] \
                or 1 == self.obs_space[0, 2 + os] == self.obs_space[0, 5 + os] == self.obs_space[0, 8 + os] \
                or 1 == self.obs_space[0, 0 + os] == self.obs_space[0, 4 + os] == self.obs_space[0, 8 + os] \
                or 1 == self.obs_space[0, 2 + os] == self.obs_space[0, 4 + os] == self.obs_space[0, 6 + os]:
            return True
        else:
            return False

    def check_if_draw(self):
        if np.sum(self.obs_space) >= 8.5:
            return True
        else:
            return False

    def pick_random_legal_move(self, player):
        if not self.check_if_draw():
            if player == 1:
                random_pool = len(self.allowed_action_space[self.allowed_action_space < 9])
                random_pick = np.random.randint(0, random_pool)
                random_action = self.allowed_action_space[random_pick]
                return random_action

            if player == 2:
                random_pool = len(self.allowed_action_space[self.allowed_action_space >= 9])
                random_pick = np.random.randint(0, random_pool)
                random_action = self.allowed_action_space[random_pick + random_pool]
                return random_action
        else:
            return False

    def render(self):
        player1_space = self.obs_space[0, :9]
        player2_space = self.obs_space[0, 9:]
        self.allowed_total_action_space=np.add(player1_space,player2_space)
        self.allowed_total_action_space[np.where(self.allowed_total_action_space==2)]=1
        player2_space = self.obs_space[0, 9:] * 2
        render_space = np.add(player1_space, player2_space)
        render_space = np.reshape(render_space, (3, 3))
        print(render_space)

        if self.game_status==1:
            print("Player 1 Wins")
        if self.game_status==2:
            print("Player 2 Wins")
        if self.game_status==3:
            print ("Draw!")
        if self.game_status==0:
            print ("Player moved", self.turn)

    def make_move(self, player, move):
        try:
            # assert (self.game_status ==0),"Game Over"
            #assert (not self.check_if_draw()), "Game Over Drawn"
            #assert (not self.check_if_game_won(player)), "Game Over Won"
            #assert (self.check_if_move_legal(player, move)), "Illegal Move"

            if player == 1:
                self.obs_space[0][move] = 1
                move_index = np.argwhere(self.allowed_action_space == move)
                self.allowed_action_space = np.delete(self.allowed_action_space, move_index)
                move_indexp2 = np.argwhere(self.allowed_action_space == move + 9)
                self.allowed_action_space = np.delete(self.allowed_action_space, move_indexp2)

            elif player == 2:
                self.obs_space[0][move] = 1
                move_index = np.argwhere(self.allowed_action_space == move)
                self.allowed_action_space = np.delete(self.allowed_action_space, move_index)
                move_indexp1 = np.argwhere(self.allowed_action_space == move - 9)
                self.allowed_action_space = np.delete(self.allowed_action_space, move_indexp1)

        except:
            print("Illegal Move")








