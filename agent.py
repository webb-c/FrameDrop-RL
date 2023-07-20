"""
RL Agent that drops unnecessary frames depending on the network state in the end device.
action : skip-length
"""
import numpy as np
import random

random.seed(42)


class Agent():
    def __init__(self, qTable=[], eps_init=1, eps_decrese=0.01, eps_min=0.1, fps=30, lr=0.1, gamma=0.9, stateNum=10):
        self.eps = eps_init
        self.eps_decrese = eps_decrese
        self.eps_min = eps_min
        self.lr = lr
        self.gamma = gamma
        self.stateNum = stateNum
        self.fps = fps
        if qTable : 
            self.qTable = qTable
        else :
            self.qTable = np.zeros((stateNum, fps))  # s, a
        self.actionSpace = list(range(fps))  # 0 to fps-1

    def get_action(self, s):
        p = random.random()
        if p < self.eps:  # exploration
            action = random.choice(self.actionSpace)
        else:  # exploitation
            qValue = self.qTable[s, :]
            action = np.argmax(qValue)
        return action

    def Q_update(self, trans):
        s, a, s_prime, r = trans
        self.qTable[s, a] = self.qTable[s, a] + self.lr * \
            (r + np.max(self.qTable[s_prime, :]) - self.qTable[s, a])
        return

    def Q_show(self):
        for s in range(self.stateNum) :
            for a in range(self.fps) :
                print(self.qTable[s][a], end="   ")
            print()
    
    def get_q_table(self) :
        return self.qTable
    
    def decrease_eps(self):
        self.eps -= self.eps_decrese
        self.eps = max(self.eps, self.eps_min)
