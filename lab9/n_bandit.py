import numpy as np
import matplotlib.pyplot as plt

class Bandit:

    def __init__(self, bandit=10, epsilon=0.1):
        self.bandit = bandit
        self.epsilon = epsilon
        self.q_star = list(np.random.randn(bandit))
        self.count = np.zeros(bandit, dtype=int)
        self.actions = []
        self.reward = []
        self.average_reward = []
        self.optimal_reward = []

    def execute_action(self, action):
        reward = self.q_star[action] + np.random.randn()
        self.greedy(reward)

        # Update utility estimate of the action we picked
        self.q_star[action] += (1 / float(self.count[action])) * (reward - self.q_star[action])



    def greedy(self, reward):
        reward_opt = np.random.choice([max(self.q_star), reward], p=[1 - self.epsilon, self.epsilon])

        # save reward and action
        self.reward.append(reward_opt)
        self.actions.append(np.where(self.q_star == max(self.q_star))[0])

    def update_average(self):
        max_q = np.where(self.q_star == max(self.q_star))[0]

        for i, (reward, reward_opt) in enumerate(zip(np.cumsum(self.reward), np.cumsum(np.array(self.actions) == max_q))):
            self.average_reward.append(reward * (i!=0) / (i + 1))
            self.optimal_reward.append(reward_opt * (i!=0) / (i + 1))



if __name__ == "__main__":

    epsilon     = 0.01  # learning parameter
    epsilons     = [0.0, 0.01, 0.1]  # learning parameter
    num_actions = 10   # number of arms on the bandit
    iterations  = 100 # number of times we repeat the experiment
    plays       = 5000 # number of plays within each experiment

    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(20, 6))

    for epsilon in epsilons:
        average_rewards = []
        average_rewards_opt = []
        for _ in range(iterations):
            bandit = Bandit(bandit=num_actions, epsilon=epsilon)
            for i in range(plays):
                arm = np.random.randint(0, num_actions)
                bandit.count[arm] += 1
                bandit.execute_action(arm)
            bandit.update_average()


            average_rewards.append(bandit.average_reward)
            average_rewards_opt.append(bandit.optimal_reward)

        average_rewards = np.mean(average_rewards, axis=0)
        average_rewards_opt = np.mean(average_rewards_opt, axis=0) * 100

        # plot
        axes[0].plot(average_rewards, label=f'$\epsilon = {epsilon}$')
        axes[1].plot(average_rewards_opt, label=f'$\epsilon = {epsilon}$')
        axes[0].set_ylabel("Average reward", fontsize=18)
        axes[1].set_ylabel("Percent optimal action", fontsize=18)

        for ax in axes:
            ax.set_xlabel("Plays", fontsize=18)
            ax.legend()
    plt.show()