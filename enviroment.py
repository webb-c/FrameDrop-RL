"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random
import cv2
import os
import win32pipe, win32file
from utils.get_state import cluster_pred, cluster_load
from utils.cal_quality import get_FFT, get_MSE
from utils.yolov5.detect import inference

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
    def __init__(self, videoPath="data/test.mp4", buffer_size=1000, fps=30, alpha=0.7, beta=10, w=5):
        self.buffer = ReplayBuffer(buffer_size)
        self.omnet = Communicator()
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
        self.omnet.init_pipe()
        self.prevA = self.fps
        self.targetA = self.fps
        _, f1 = self.cap.read()
        _, f2 = self.cap.read()
        self.frameList = [f1, f2]
        self.processList = [f2]
        self.prev_frame = self.frameList[-2]
        self.frame = self.frameList[-1]
        self.net = self._get_sNet()
        self.state = cluster_pred(
            get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net, self.model)
        self._detect()
        return self.state

    def step(self, action):
        # skipping
        guided = False
        start = False
        for a in range(action+1):
            ret, temp = self.cap.read()
            if not ret:
                return _, True
            if len(self.frameList) >= self.fps:
                # call OMNeT++
                guided = True
                newA = self.omnet.get_omnet_message() # request : pipe return A(t) -> wait OMNET
                start = self._triggered_by_guide(newA, temp, action, a)
                self.omnet.send_ommnet_message()            # request : wake up OMNet
            else:
                self.frameList.append(temp)

        if not start : 
            self.prev_frame = self.frameList[-2]
            self.frame = self.frameList[-1]
            self.processList.append(self.frame)
        
        if not guided:
            # prev_trans append s_prime
            if len(self.transList) > 0 :
                self.transList[-1].append(self.state)
            # curr_trans (s, a)
            self.transList.append((self.state, action))
        
        self.net = self._get_sNet()
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
        self.prevA = self.targetA
        self.targetA = newA
        self.net = self._get_sNet()
        self.state = cluster_pred(
            get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net, self.model)
        # curr_trans append s_prime
        self.transList[-1].append(self.state)
        # reward
        self._get_reward()
        self.buffer.put(self.transList)
        self.transList = []
        # subTask 2
        na = action-a-1
        if na >= 0:
            # curr_trans (s, a)
            self.transList = [(self.state, na)]
            return False
        return True
    
    def _get_sNet(self):
        return (self.targetA - len(self.processList))/(self.fps + 1 - len(self.frameList))
    
    def _detect(self):
        command = ["--weights", "yolov5s6.pt", "--source", "../../"+self.videoPath, "--save-txt", "--save_conf", "--nosave"]
        inference(command) # cls, *xywh, conf
        dirPath = "utils/runs/detect/exp"
        fileList =  os.listdir(dirPath)
        objNumList = []
        for fileName in fileList :
            filePath = dirPath+"/"+fileName
            with open(filePath, 'r') as file :
                lines = file.readlines()
                objNumList.append(len(lines))
        print(objNumList)
        
    def _get_reward(self):
        length = len(self.transList)
        for i in range(length):
            s, a = self.transList[i]
            # request addition (YOLO -> self.frameList detect)
            self.transList[i].append(r)
        return

class Communicator(Exception):
    def __init__(self, pipeName, buffer_size):
        self.pipeName = pipeName
        self.buffer_size = buffer_size
        self.pipe = self.init_pipe(self.pipe_name, self.buffer_size)

    def send_omnet_message(self, msg):
        win32file.WriteFile(self.pipe, msg.encode('utf-8'))  # wait unil complete reward cal & a(t)
        
    def get_omnet_message(self):
        response_byte = win32file.ReadFile(self.pipe, self.buffer_size)
        response_str = response_byte[1].decode('utf-8')
        return response_str

    def close_pipe(self):
        win32file.CloseHandle(self.pipe)

    def init_pipe(self, pipe_name, buffer_size):
        pipe = None
        try:
            pipe = win32pipe.CreateNamedPipe(
                pipe_name,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1,
                buffer_size,
                buffer_size,
                0,
                None
            )
        except:
            print("except : pipe error")
            return -1

        win32pipe.ConnectNamedPipe(pipe, None)

        return pipe

# test
env = FrameEnv()
env._detect()