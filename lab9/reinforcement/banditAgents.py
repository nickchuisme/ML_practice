from pacman import Directions
from game import Agent
import game
import util
import numpy as np

class BanditAgent(Agent):

    def __init__(self, arms=5, epsilon=0.1, temperature=0.1, method='softmax'):
        self.arms = arms
        self.epsilon = epsilon
        self.temperature = temperature
        self.method = method

        self.q_star = list(np.zeros(arms, dtype=float))
        self.count = np.zeros(arms, dtype=int)
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

        if self.method == 'greedy':
            return self.greedy(legal)
        elif self.method == 'softmax':
            return self.softmax(legal)

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
            action_idx = list(np.where(np.array(legal_of_q) == max(legal_of_q)))[0][0]
            return legals[action_idx]
        else:
            return np.random.choice(legals)

    def softmax(self, legals):
        temperature = self.temperature
        legal_of_q = [self.q_star[self.direction[legal]] for legal in legals]
        z = np.sum(np.exp(np.array(legal_of_q) / temperature))
        probs = np.exp(np.array(legal_of_q) / temperature) / z

        return np.random.choice(legals, p=probs)