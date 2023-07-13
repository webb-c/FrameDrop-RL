"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random

random.seed(42)
class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)
        
    def put(self, transition) :
        self.buffer.append(transition)
    
    def get(self) :
        transition = random.sample(self.buffer, 1)  # buffer size = 1
        return transition
    
    def size(self) :
        return len(self.buffer)
    
class FrameEnv():
    def __init__(self, fps=30, alpha=0.7, beta=10, w=5):
        self.fps = fps
        self.alpha = alpha
        self.alpha = beta
        self.w = w
        self.actionSpace = list(range(fps+1)) # 0 to fps
        
    def reset(self) : 
        return
    
    def step(self, action) :
        return
    
    def get_reward(self) :
        return