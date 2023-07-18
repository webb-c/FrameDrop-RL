"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random
import cv2
from utils.get_state import cluster_pred, cluster_load
from utils.cal_quality import get_FFT, get_MSE
import utils.yolov5.detect

random.seed(42)


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def put(self, transList):
        for trans in transList:
            self.buffer.append(trans)

    def get(self):
        trans = random.sample(self.buffer, 1)  # batch size = 1
        return trans

    def size(self):
        return len(self.buffer)


class FrameEnv():
    def __init__(self, videoPath, buffer_size=1000, fps=30, alpha=0.7, beta=10, w=5):
        self.buffer = ReplayBuffer(buffer_size)
        self.videoPath = videoPath
        self.fps = fps
        # hyper-parameter
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.model = cluster_load()
        # state
        self.reset()

    def reset(self):
        self.transList = []
        self.cap = cv2.VideoCapture(self.videoPath)
        self.prevA = self.fps
        self.targetA = self.fps
        self.net = 0
        _, f1 = self.cap.read()
        _, f2 = self.cap.read()
        self.frameList = [f1, f2]
        self.processList = [f1, f2]
        self.prev_frame = self.frameList[-2]
        self.frame = self.frameList[-1]
        self.state = cluster_pred(
            get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net, self.model)
        return self.state

    def step(self, action, newA):
        # skipping
        guided = False
        for a in range(action+1):
            ret, temp = self.cap.read()
            if not ret:
                return _, True
            if len(self.frameList) >= self.fps:
                guided = True
                # call OMNeT++
                self._triggered_by_guide(newA, temp, action, a)
            else:
                self.frameList.append(temp)

        if not guided:
            self.prev_frame = self.frameList[-2]
            self.frame = self.frameList[-1]
            self.processList.append(self.frame)
            # prev_trans append s_prime
            if len(self.transList) > 0 :
                self.transList[-1].append(self.state)
            # curr_trans (s, a)
            self.transList.append((self.state, action))
            # new_state (curr_trans s_prime)
            self.net = max(self.net - action, 0)
            
        self.state = cluster_pred(
            get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net, self.model)

        return self.state, False

    def _triggered_by_guide(self, newA, temp, action, a):
        # subTask 1
        # prev_trans append s_prime
        if len(self.transList) > 0 :
            self.transList[-1].append(self.state)
        # curr_trans (s, a)
        self.transList.append((self.state, a))
        # new state (curr_trans s_prime)
        self.prev_frame = self.frameList[-1]
        self.frame = temp
        self.frameList = [self.frame]
        self.processList = [self.frame]
        self.net = max(self.fps - self.targetA, 0)
        self.state = cluster_pred(
            get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net, self.model)
        # curr_trans append s_prime
        self.transList[-1].append(self.state)
        # reward
        self._get_reward()
        self.buffer.put(self.transList)
        # init
        self.transList = []
        self.prevA = self.targetA
        self.targetA = newA
        # subTask 2
        na = action-a-1
        if na >= 0:
            # curr_trans (s, a)
            self.transList = [(self.state, na)]
            # new  network state
            self.net = max(self.fps - self.targetA - na, 0)
        return

    def _get_reward(self):
        length = len(self.transList)
        for i in range(length):
            s, a = self.transList[i]
            # request addition (YOLO -> self.frameList detect)
            self.transList[i].append(r)
        return
