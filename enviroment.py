"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random
import cv2
from utils.get_state import cluster_pred
from utils.cal_quality import get_FFT, get_MSE
import utils.yolov5.detect

random.seed(42)


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def put(self, transition):
        self.buffer.append(transition)

    def get(self):
        transition = random.sample(self.buffer, 1)  # buffer size = 1
        return transition

    def size(self):
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
        self.state = cluster_pred(
            get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net)

    def reset(self):
        self.cap = cv2.VideoCapture(self.videoPath)
        self.targetA = self.fps
        self.net = 0
        _, self.prev_frame = self.cap.read()
        _, self.frame = self.cap.read()

    def step(self, action, isGuided, targetA):
        for _ in range(action-1):
            self.cap.read()
        _, self.prev_frame = self.cap.read()
        ret, self.frame = self.cap.read()
        if isGuided:
            self.targetA = targetA
            self.net = max(self.fps - self.targetA, 0)
        else:
            self.net = max(self.net - action, 0)
        self.state = cluster_pred(
            get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net)
        return self.state

    def get_reward(self):
        return
