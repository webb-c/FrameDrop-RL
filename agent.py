"""
RL Agent that drops unnecessary frames depending on the network state in the end device.
action : skip-length
"""
import numpy as np
import random
import copy

random.seed(42)
INF = float("inf")

class Agent():
    def __init__(self, qTable=[], eps_init=1, eps_decrese=0.01, eps_min=0.1, fps=30, lr=0.1, gamma=0.9, stateNum=15, isRun=False, masking=True, isContinue=False):
        self.eps = eps_init
        self.eps_decrese = eps_decrese
        self.eps_min = eps_min
        self.lr = lr
        self.gamma = gamma
        self.stateNum = stateNum
        self.fps = fps
        self.masking = masking
        if len(qTable) > 0 and (isContinue or isRun) : 
            self.qTable = qTable
        else :
            self.qTable = np.zeros((stateNum, fps+1))  # s, a
        self.actionSpace = list(range(fps+1))  # 0 to fps
        self.isRun = isRun
        self.isFirst = True
        ##### NEW! ######
        # self.optProb = optProb
    
    def normal_pdf(x, mu, sigma_sq):
        return (1 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(-(x - mu)**2 / (2 * sigma_sq))

    # TODO
    def conditional_action(self, requireskip, actionList=[]) :
        optimal_action = requireskip
        # action 확률
        
        # Q value 고려
        if len(actionList) != 0 :
            
        
        ##### 선택 #####
        p = random.random()
        action = optimal_action
        return action

    # requireSkip = fps - A(t)
    def get_action(self, s, requireskip, randAction=True):
        if self.isFirst and requireskip == self.fps :
            requireskip -= 1
            self.isFirst = False
        # Inference #TODO
        if self.isRun :
            temp = copy.deepcopy(self.qTable[s, :])
            temp_vec = temp.flatten()
            while True :
                action = np.argmax(temp_vec)
                if action >= requireskip :
                    break
                temp_vec[action] = (-1)*INF
        # Training 
        else :
            if randAction :
                action = random.choice(self.actionSpace[:])
            else : 
                p = random.random()
                # exploration
                if p < self.eps :
                    if self.masking : 
                        # masked ---- soft-constraint ---- 
                        action = self.conditional_action(requireskip)  # 100% probabiltiy 
                        
                    else :  
                        action = random.choice(self.actionSpace[:])
                # argmax
                else:
                    temp = copy.deepcopy(self.qTable[s, :])
                    temp_vec = temp.flatten()
                    if self.masking : 
                        # masked ---- soft-constraint ----
                        # while True :
                        #     action = np.argmax(temp_vec)
                        #     if action >= requireskip :
                        #         break
                        #     temp_vec[action] = (-1)*INF
                        action = self.conditional_action(requireskip, actionList=temp_vec)
                    else :
                        action = np.argmax(temp_vec)
        return action

    def Q_update(self, trans):
        requireskip, s, a, s_prime, r = trans
        self.qTable[s, a] = self.qTable[s, a] + self.lr * \
            (r + np.max(self.qTable[s_prime, requireskip:]) - self.qTable[s, a])
        return

    def Q_show(self):
        for s in range(self.stateNum) :
            for a in range(self.fps+1) :
                print(round(self.qTable[s][a], 2), end="   ")
            print()
    
    def get_q_table(self) :
        return self.qTable
    
    def decrease_eps(self):
        self.eps -= self.eps_decrese
        self.eps = max(self.eps, self.eps_min)
        print("epsilon :", self.eps)