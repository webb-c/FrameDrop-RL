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
    def __init__(self, qTable=[], eps_init=1, eps_decrese=0.01, eps_min=0.1, fps=30, lr=0.1, gamma=0.9, stateNum=15, softWeight=0.9, isRun=False, masking=True, isContinue=False, isSoft=False):
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
        # ********** NEW **********
        self.isSoft = isSoft
        self.softWeight = softWeight
    
    # gaussian distribution for soft=constraint
    def gaussian(self, x, mean, std):
        return (1.0 / np.sqrt(2 * np.pi * std**2)) * np.exp(-((x - mean)**2) / (2 * std**2))
    
    def sample_gaussian(self, mean, std=1.2, lower_bound=0, upper_bound=30):
        x = np.arange(lower_bound, upper_bound+1)
        y = self.gaussian(x, mean, std)
        sampled_value = np.random.choice(x, p=y/np.sum(y))
        return sampled_value
    
    def discrete_gaussian_prob(self, mean, std=1.2, lower_bound=0, upper_bound=30):
        x = np.arange(lower_bound, upper_bound + 1)
        y = (1.0 / np.sqrt(2 * np.pi * std ** 2)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return y / np.sum(y)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    # requireSkip = fps - A(t)
    def get_action(self, s, requireskip, randAction=True):
        if self.isFirst and requireskip == self.fps :
            requireskip -= 1
            self.isFirst = False
        if self.isRun :
            temp = copy.deepcopy(self.qTable[s, :])
            temp_vec = temp.flatten()
            while True :
                action = np.argmax(temp_vec)
                if action >= requireskip :
                    break
                temp_vec[action] = (-1)*INF
        else :
            if randAction :
                action = random.choice(self.actionSpace[:])
            # masked
            else : 
                p = random.random()
                # exploration
                if p < self.eps :  
                    # ***** hard-soft *****
                    if self.masking : 
                        if self.isSoft:
                            action = self.sample_gaussian(mean=requireskip, std=5)  #TODO std parameter?
                        else :
                            action = random.choice(self.actionSpace[requireskip:])
                    else : 
                        action = random.choice(self.actionSpace[:])
                # greedy
                else:  
                    temp = copy.deepcopy(self.qTable[s, :])
                    temp_vec = temp.flatten()
                    if self.masking : 
                        # TODO soft-constraint with argmax
                        if self.isSoft :
                            gaussian_prob = self.discrete_gaussian_prob(mean=requireskip, std=self.softWeight)
                            """
                            ##### not-softmax #####
                            min_val = np.min(temp_vec)
                            max_val = np.max(temp_vec)
                            Q_prob = (temp_vec - min_val) / (max_val - min_val)
                            """
                            ##### using softmax #####
                            Q_prob = self.softmax(np.array(temp_vec))
                            """
                            ##### using + #####
                            combined_prob = (1-self.softWeight)*Q_prob + self.softWeight*gaussian_prob
                            action = np.random.choice(np.arange(0, 31), p=combined_prob / np.sum(combined_prob))
                            """
                            ##### using joint #####
                            combined_prob = [g * q for g, q in zip(gaussian_prob, Q_prob)]
                            # action = np.random.choice(np.arange(0, 31), p=np.array(combined_prob)/np.sum(combined_prob))
                            ###### using hardconstraint #####
                            filtered_indices = [i for i, p in enumerate(combined_prob) if p > self.softWeight]
                            if filtered_indices:
                                Q_values = [temp_vec[i] for i in filtered_indices]
                                max_Q_value = max(Q_values)
                                action = filtered_indices[Q_values.index(max_Q_value)]
                            else :
                                action = np.random.choice(np.arange(0, 31), p=np.array(combined_prob)/np.sum(combined_prob))
                            
                        else :
                            while True :
                                action = np.argmax(temp_vec)
                                if action >= requireskip :
                                    break
                                temp_vec[action] = (-1)*INF
                    else :
                        action = np.argmax(temp_vec)
        return action

    def Q_update(self, trans):
        #TODO
        requireskip, s, a, s_prime, r = trans
        self.qTable[s, a] = self.qTable[s, a] + self.lr * \
            (r + np.max(self.qTable[s_prime, :]) - self.qTable[s, a])
        """
        self.qTable[s, a] = self.qTable[s, a] + self.lr * \
            (r + np.max(self.qTable[s_prime, requireskip:]) - self.qTable[s, a])
        filtered_indices, s, a, s_prime, r = trans
        temp = copy.deepcopy(self.qTable[s_prime, :])
        temp_vec = temp.flatten()
        if filtered_indices:
            Q_values = [temp_vec[i] for i in filtered_indices]
            optimal = max(Q_values)
        else :
            optimal =  np.max(self.qTable[s_prime, :])

        self.qTable[s, a] = self.qTable[s, a] + self.lr * \
            (r + optimal - self.qTable[s, a])
        """
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