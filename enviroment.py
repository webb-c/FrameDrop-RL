"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random
import cv2
from utils.get_state import cluster_pred
import utils.cal_quality
import utils.yolov5.detect

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
    def __init__(self, videoPath, fps=30, alpha=0.7, beta=10, w=5):
        self.videoPath = videoPath
        self.fps = fps
        # hyper-parameter
        self.alpha = alpha
        self.beta = beta
        self.w = w
        # state
        self.reset()
        # self.state = clueter_pred()
        
    def reset(self) :
        self.cap = cv2.VideoCapture(self.videoPath)
        _, self.prev_frame = self.cap.read()
        _, self.frame = self.cap.read()
        
    def step(self, action) :
        for _ in range(action-1) :
            self.cap.read()
        _, self.prev_frame = self.cap.read()
        ret, self.frame = self.cap.read()
        # self.state = cluster_pred()
        return self.state
    
    def get_reward(self) :
        return