# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        # basic features from original state class
        self.score = state.getScore()
        self.legal = state.getLegalPacmanActions()
        if 'Stop' in self.legal:
            self.legal.remove('Stop')

        pac_position = state.getPacmanPosition()
        ghost_position = state.getGhostPositions()
        closet_ghost_pos = ghost_position[np.argmin([self.manhattan_distance(pac_position, g_pos) for g_pos in ghost_position])]
        food = state.getFood()
        # create "condition" feature to save information of each state, there are:
        # - pacman's position
        # - position of the closet ghost (if there are multiple ghosts)
        # - food's state

        # if manhattan distance is greater than 2, do not add ghost's position into condition feature
        if self.manhattan_distance(pac_position, closet_ghost_pos) > 2:
            self.condition = (pac_position, food)
        else:
            self.condition = (pac_position, closet_ghost_pos, food)

    def manhattan_distance(self, pos1, pos2):
        # calculate manhattan distance
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        self.NE = 12
        self.R_PLUS = 7

        self.q_table = dict()
        self.actions_count = dict()
        self.prev_state = None
        self.prev_action = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Compute reward with previous and current state

        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return endState.score - startState.score

    def initQValue(self,
                   state: GameStateFeatures):
        """
        initial Q-table

        Args:
            state: A given state
        """

        for action in state.legal:
            q_sa = (state.condition, action)
            if q_sa not in self.q_table:
                self.q_table[q_sa] = 0.0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Get estimation from state and action

        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        return self.q_table[(state.condition, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Get the maximum estimation

        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        return max([self.q_table[(state.condition, action)] for action in state.legal])

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        q = self.getQValue(state, action)
        q_max = self.maxQValue(nextState)
        self.q_table[(state.condition, action)] = q + self.alpha * (reward + self.gamma * q_max - q)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        q_sa = (state.condition, action)
        if q_sa not in self.actions_count:
            self.actions_count[q_sa] = 1
        else:
            self.actions_count[q_sa] += 1

    def save_state_action(self,
                          state: GameStateFeatures,
                          action: Directions):
        """
        update previous state and action

        Args:
            state: Starting state
            action: Action taken
        """
        self.prev_state = state
        self.prev_action = action

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Obtain the record of the count of appearence of each state
        if the state does not exist, return 0

        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        try:
            return self.actions_count[(state.condition, action)]
        except:
            return 0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        if counts < self.NE:
            return self.R_PLUS
        else: 
            return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        # initial Q(s,a)
        self.initQValue(stateFeatures)

        if self.prev_state:
            # calculate reward and update Q(s,a)
            reward = self.computeReward(self.prev_state, stateFeatures)
            self.updateCount(self.prev_state, self.prev_action)
            self.learn(state=self.prev_state, action=self.prev_action, reward=reward, nextState=stateFeatures)

        # Now pick what action to take.
        # The current code shows how to do that but just makes the choice randomly.

        # choose a exploration or exploitation with epsilon
        exploration = np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])
        if exploration:
            # get utilities and counts
            utility = np.vectorize(self.getQValue)(stateFeatures, legal)
            counts = np.vectorize(self.getCount)(stateFeatures, legal)

            # find utilities via exploration function
            explore_value = np.vectorize(self.explorationFn)(utility, counts)
            # exploit action with the maximum utility
            action = legal[np.argmax(explore_value)]
        else:
            # exploit action with the maximum utility
            action = legal[np.argmax(np.vectorize(self.getQValue)(stateFeatures, legal))]

        # update previous state
        self.save_state_action(stateFeatures, action)
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """

        # update final state
        reward = state.getScore() - self.prev_state.score
        self.updateCount(self.prev_state, self.prev_action)
        q = self.getQValue(self.prev_state, self.prev_action)
        q_max = 0
        self.q_table[(self.prev_state.condition, self.prev_action)] = q + self.alpha * (reward + self.gamma * q_max - q)

        # reset variable
        self.prev_state = None
        self.prev_action = None

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
