"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random
import cv2
import os
import math
import win32pipe, win32file
from utils.get_state import cluster_pred, cluster_load, cluster_init
from utils.cal_quality import get_FFT, get_MSE
from utils.cal_F1 import get_F1
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
    def __init__(self, videoPath="data/test.mp4", buffer_size=1000, fps=30, alpha=0.7, beta=10, w=5, isClusterexist=False):
        self.isClusterexist = isClusterexist
        self.buffer = ReplayBuffer(buffer_size)
        self.data = collections.deque(maxlen=1000)
        self.omnet = Communicator("\\\\.\\pipe\\frame_drop_rl", 200000)
        self.videoPath = videoPath
        self.fps = fps
        # hyper-parameter
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.model = cluster_init(k=self.fps)
        if self.isClusterexist :  
            self.model = cluster_load()
        # self._detect()
        # state
        self.reset()

    def reset(self, isClusterexist=False):
        self.isClusterexist = isClusterexist
        self.cap = cv2.VideoCapture(self.videoPath)
        self.idx = 0
        self.prevA = self.fps
        self.targetA = self.fps
        _, f1 = self.cap.read()
        _, f2 = self.cap.read()
        self.frameList = [f1, f2]
        self.processList = [f2]
        self.prev_frame = self.frameList[-2]
        self.frame = self.frameList[-1]
        self.net = self._get_sNet()
        self.originState = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net]
        self.state = 0
        if self.isClusterexist :
            self.transList = []
            self.state = cluster_pred(self.originState, self.model)
        return self.state

    def step(self, action):
        # skipping
        guided = False
        start = False
        for a in range(action+1):
            ret, temp = self.cap.read()
            if not ret:
                return -1, True
            if len(self.frameList) >= self.fps:
                # call OMNeT++
                guided = True
                print("request guide")
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("reward") 
                ratioA = float(self.omnet.get_omnet_message())
                newA = math.floor(ratioA*(self.fps))
                print("new A :", newA)
                start = self._triggered_by_guide(newA, temp, action, a)
            else:
                self.frameList.append(temp)

        if not start : 
            self.prev_frame = self.frameList[-2]
            self.frame = self.frameList[-1]
            self.processList.append(self.frame)
        
        if not guided:
            # prev_trans append s_prime
            if self.isClusterexist :
                if len(self.transList) > 0 :
                    self.transList[-1].append(self.state)
                # curr_trans (s, a)
                self.transList.append((self.state, action))
            print("state: ",self.originState,"action: ", action)
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message("action")
            self.omnet.send_omnet_message(str((action+1)/self.fps))
            self.data.append(self.originState)
        
        self.net = self._get_sNet()
        self.originState = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net]
        if self.isClusterexist :
            self.state = cluster_pred(self.originState, self.model)
        return self.state, False

    def _triggered_by_guide(self, newA, temp, action, a):
        # subTask 1
        # prev_trans append s_prime
        if self.isClusterexist :
            if len(self.transList) > 0 :
                self.transList[-1].append(self.state)  
            # curr_trans (s, a)
            self.transList.append((self.state, a))
        print("state: ",self.originState,"action: ", a)
        self.omnet.get_omnet_message()
        self.omnet.send_omnet_message("action")
        self.omnet.send_omnet_message(str((a+1)/self.fps))
        self.data.append(self.originState)
        # new state (curr_trans s_prime)
        self.prev_frame = self.frameList[-1]
        self.frame = temp
        self.frameList = [self.frame]
        self.processList = [self.frame]
        self.prevA = self.targetA
        self.targetA = newA
        self.net = self._get_sNet()
        self.originState = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net]
        if self.isClusterexist :
            self.state = cluster_pred(self.originState, self.model)
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
            if self.isClusterexist : 
                self.transList = [(self.state, na)]
            print("state: ",self.originState,"action: ", na)
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message("action")
            self.omnet.send_omnet_message(str((na+1)/self.fps))
            return False
        return True
    
    def _get_sNet(self):
        return (self.targetA - len(self.processList))/(self.fps + 1 - len(self.frameList))
    
    def _detect(self, exist=False):
        command = ["--weights", "models/yolov5s6.pt", "--source", self.videoPath, "--save-txt", "--save-conf", "--nosave"]
        if not exist : 
            inference(command) # cls, *xywh, conf
        self.resultPath = "utils/yolov5/runs/detect/exp/labels"
        fileList =  os.listdir(self.resultPath)
        self.objNumList = []  # 물체개수
        for fileName in fileList :
            filePath = self.resultPath+"/"+fileName
            with open(filePath, 'r') as file :
                lines = file.readlines()
                self.objNumList.append(len(lines))
        
    def _get_reward(self):
        length = len(self.transList)
        # get importance
        self.iList = []
        for f in range(self.fps) :
            sIdx = max(0, f-self.w)
            eIdx = max(self.fps, f+self.w+1)
            maxNum = max(self.objNumList[sIdx:eIdx])
            inv_importance = (1 - self.objNumList[i] / maxNum) if maxNum != 0 else 0
            self.iList.append(inv_importance)
        del self.objNumList[:self.fps] # remove already using
        A_diff = self.targetA - self.prevA
        for i in range(length):
            s, a, s_prime = self.transList[i]
            # request addition (YOLO -> self.frameList detect)
            r_blur = sum(self.iList[self.idx:self.idx+a+1]) / a
            self.F1List = []
            refFrame = self.resultPath+"/test_"+str(self.idx+1)+".txt"
            for k in range(1, a+1) :
                skipFrame = self.resultPath+"/test_"+str(self.idx+k+1)+".txt"
                self.F1List.append(1 - get_F1(refFrame, skipFrame))
            r_dup = sum(self.F1List)
            if A_diff >= 0 :
                r_net = (a/self.fps)*(A_diff)
            else :
                r_net = self.beta * ((self.fps - a) / self.fps) * (A_diff)
            r = (1 - self.alpha) * (r_blur - r_dup) + self.alpha * (r_net)
            self.transList[i].append(r)
            self.idx += a+1
        return

class Communicator(Exception):
    def __init__(self, pipeName, buffer_size):
        self.pipeName = pipeName
        self.buffer_size = buffer_size
        self.pipe = self.init_pipe(self.pipeName, self.buffer_size)

    def send_omnet_message(self, msg):
        win32file.WriteFile(self.pipe, msg.encode('utf-8'))  # wait unil complete reward cal & a(t)
        
    def get_omnet_message(self):
        response_byte = win32file.ReadFile(self.pipe, self.buffer_size)
        response_str = response_byte[1].decode('utf-8')
        return response_str

    def close_pipe(self):
        win32file.CloseHandle(self.pipe)

    def init_pipe(self, pipeName, buffer_size):
        pipe = None
        print("waiting connect OMNeT++ ...")
        try:
            pipe = win32pipe.CreateNamedPipe(
                pipeName,
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
def main():
    env = FrameEnv()
    
if __name__ == "__main__" :
    main()