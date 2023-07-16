"""
RL Agent that drops unnecessary frames depending on the network state in the end device.
action : skip-length
"""
import numpy as np
import random
from enviroment import FrameEnv

random.seed(42)


class Agent():
    def __init__(self, eps=0.9, fps=30):
        self.eps = eps
        self.Qtable = np.zeros((30, 30))  # s, a
        self.actionSpace = list(range(fps+1))  # 0 to fps

    def get_action(self, s):
        s1, s2 = s
        p = random.random()
        if p < self.eps:  # exploration
            action = random.sample(self.actionSpace, 1)
        else:  # exploitation
            Qvalue = self.Qtable[s1, s2, :]
            action = np.argmax(Qvalue)
        return action

    def Q_update(self):
        return

    def decrese_eps(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.1)
