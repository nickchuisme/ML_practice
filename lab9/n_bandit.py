import numpy as np
import matplotlib.pyplot as plt

class Bandit:

    def __init__(self, arms=10, epsilon=0.1, tau=10000, temperature=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.q_star = list(np.random.randn(arms))
        self.count = np.zeros(arms, dtype=int)
        self.actions = []
        self.reward = []
        self.average_reward = []
        self.optimal_reward = []
        self.tau = tau
        self.temperature = temperature

    def count_update(self, idx):
        self.count[idx] += 1

    def reward_update(self, idx):
        reward = self.q_star[idx] + np.random.randn()

        # Update utility estimate of the action we picked
        self.q_star[idx] += (1 / float(self.count[idx])) * (reward - self.q_star[idx])
        self.reward.append(self.q_star[idx])

    def execute_action(self, selection='greedy'):
        if selection == 'greedy':
            reward_idx = self.greedy()
        elif selection == 'softmax':
            reward_idx = self.softmax()

        self.actions.append(reward_idx)
        self.count_update(reward_idx)
        self.reward_update(reward_idx)

    def greedy(self):
        exploitation = np.random.choice([True, False], p=[1 - self.epsilon, self.epsilon])
        if exploitation:
            return np.where(self.q_star == max(self.q_star))[0][0]
        else:
            return np.random.randint(0, self.arms)

    def softmax(self):
        # temperature = self.tau / np.sum(self.count + 1)
        temperature = self.temperature
        z = np.sum(np.exp(np.array(self.q_star) / temperature))
        probs = np.exp(np.array(self.q_star) / temperature) / z

        arm = np.random.choice(self.q_star, p=probs)
        return np.where(self.q_star == arm)[0][0]

    def update_average(self):
        max_q = np.where(self.q_star == max(self.q_star))[0]

        for i, (reward, reward_opt) in enumerate(zip(np.cumsum(self.reward), np.cumsum(np.array(self.actions) == max_q))):
            self.average_reward.append(reward * (i!=0) / (i + 1))
            self.optimal_reward.append(reward_opt * (i!=0) / (i + 1))


if __name__ == "__main__":
    epsilon     = 0.01  # learning parameter
    epsilons     = [0.0, 0.01, 0.1]  # learning parameter
    temperatures = [0.1, 0.3, 0.5]
    num_actions = 10   # number of arms on the bandit
    iterations  = 100 # number of times we repeat the experiment
    plays       = 5000 # number of plays within each experiment
    method = 'softmax'

    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(20, 6))

    if method == 'greedy':
        for epsilon in epsilons:
            average_rewards = []
            average_rewards_opt = []

            for _ in range(iterations):
                bandit = Bandit(arms=num_actions, epsilon=epsilon)

                for i in range(plays):
                    bandit.execute_action(selection=method)

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
    
    elif method == 'softmax':
        for temperature in temperatures:
            average_rewards = []
            average_rewards_opt = []

            for _ in range(iterations):
                bandit = Bandit(arms=num_actions, temperature=temperature)

                for i in range(plays):
                    bandit.execute_action(selection=method)

                bandit.update_average()
                average_rewards.append(bandit.average_reward)
                average_rewards_opt.append(bandit.optimal_reward)

            average_rewards = np.mean(average_rewards, axis=0)
            average_rewards_opt = np.mean(average_rewards_opt, axis=0) * 100

            # plot
            axes[0].plot(average_rewards, label=f'$temperature = {temperature}$')
            axes[1].plot(average_rewards_opt, label=f'$temperature = {temperature}$')
            axes[0].set_ylabel("Average reward", fontsize=18)
            axes[1].set_ylabel("Percent optimal action", fontsize=18)

            for ax in axes:
                ax.set_xlabel("Plays", fontsize=18)
                ax.legend()
    plt.show()