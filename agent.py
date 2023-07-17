"""
RL Agent that drops unnecessary frames depending on the network state in the end device.
action : skip-length
"""
import numpy as np
import random

random.seed(42)


class Agent():
    def __init__(self, eps=1, fps=30, lr=0.1, gamma=0.9):
        self.eps = eps
        self.lr = lr
        self.gamma = gamma
        self.qTable = np.zeros((30, 30))  # s, a
        self.actionSpace = list(range(fps))  # 0 to fps-1

    def get_action(self, s):
        p = random.random()
        if p < self.eps:  # exploration
            action = random.sample(self.actionSpace, 1)
        else:  # exploitation
            qValue = self.qTable[s, :]
            action = np.argmax(qValue)
        return action

    def Q_update(self, trans):
        s, a, s_prime, r = trans
        self.qTable[s, a] = self.qTable[s, a] + self.lr * \
            (r + np.max(self.qTable[s_prime, :]) - self.qTable[s, a])
        return

    def decrease_eps(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.1)
