from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np

class BanditAgent(Agent):

    def __init__(self, bandit=5, epsilon=0.1):
        self.bandit = bandit
        self.epsilon = epsilon
        self.q_star = list(np.zeros(bandit, dtype=float))
        self.count = np.zeros(bandit, dtype=int)

        self.score = 0
        self.direction = {
            'East': 0,
            'West': 1,
            'South': 2,
            'North': 3,
            'Stop': 4
        }

    def getAction(self, state):
        self.update_reward(state)

        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        return self.greedy(legal)

    def update_reward(self, state):
        # get reward of previous action
        reward = state.getScore() - self.score
        self.score = state.getScore()

        # get previous action
        last_action = self.direction[state.getPacmanState().configuration.direction]

        # calculate step
        self.count[last_action] += 1

        # update utility
        self.q_star[last_action] += (1 / float(self.count[last_action])) * (reward - self.q_star[last_action])

    def greedy(self, legals):
        # exploration or exploitation
        exploitation = np.random.choice([True, False], p=[1 - self.epsilon, self.epsilon])
        if exploitation:
            legal_of_q = [self.q_star[self.direction[legal]] for legal in legals]
            action_idx = list(np.where(np.array(legal_of_q) == max(legal_of_q)))[0]
            return legals[action_idx[0]]
        else:
            return np.random.choice(legals)
