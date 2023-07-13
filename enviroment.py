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