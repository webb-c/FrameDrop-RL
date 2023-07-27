"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random
import cv2
import os
import math
import numpy as np
import win32pipe, win32file
from utils.get_state import cluster_pred, cluster_load, cluster_init
from utils.cal_quality import get_FFT, get_MSE
from utils.cal_F1 import get_F1
from utils.yolov5.detect import inference

random.seed(42)


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def put(self, trans):
        self.buffer.append(trans)
            
    def get(self):
        trans = random.choice(self.buffer)  # batch size = 1
        return trans

    def size(self):
        return len(self.buffer)


class FrameEnv():
    def __init__(self, videoName, videoPath, resultPath, clusterPath, data_maxlen=10000, replayBuffer_maxlen=10000, fps=30, w=5, stateNum=15, isDetectionexist=True, isClusterexist=False, isRun=False, outVideoPath="./output.mp4"):
        self.isDetectionexist = isDetectionexist
        self.isClusterexist = isClusterexist
        self.buffer = ReplayBuffer(replayBuffer_maxlen)
        self.data = collections.deque(maxlen=data_maxlen)
        self.omnet = Communicator("\\\\.\\pipe\\frame_drop_rl", 200000)
        self.videoName = videoName
        self.videoPath = videoPath
        self.resultPath = resultPath
        self.fps = fps
        self.isRun = isRun
        self.clusterPath = clusterPath
        self.capIdx = 0
        # hyper-parameter
        self.w = w
        self.model = cluster_init(k=stateNum)
        if self.isClusterexist :  
            self.model = cluster_load(self.clusterPath)
        if not self.isRun :
            self._detect(self.isDetectionexist)
        # record
        self.ASum = 0
        self.aSum = 0
        self.skipTime = 0
        # state
        self.reset()
        if self.isRun :
            self.processedFrameList = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.outVideoPath = outVideoPath
            self.out = cv2.VideoWriter(outVideoPath, fourcc, self.fps, (self.frame.shape[1], self.frame.shape[0]))
            
    def reset(self, isClusterexist=False, showLog=False) :
        self.reward_sum = 0 # r_dup, r_blur, r_net, r_total
        self.skipTime = 0
        self.ASum = 0
        self.aSum = 0
        self.iList = []
        self.isClusterexist = isClusterexist
        self.showLog = showLog
        self.cap = cv2.VideoCapture(self.videoPath)
        self.curFrameIdx = 0
        self.prevA = self.fps
        self.targetA = self.fps
        _, f1 = self.cap.read()
        self.capIdx = 0
        self.frameList = [f1]
        self.processList = []
        self.prev_frame = f1
        self.frame = f1
        if self.isRun :
            self.originState = [0, get_FFT(self.frame)]
        else :
            self.originState = [0, self.FFTList[self.curFrameIdx]]
        self.state = 0
        if self.isClusterexist :
            self.transList = []
            self.state = cluster_pred(self.originState, self.model)
        if self.showLog :   
            self.logList = []
        return self.state

    def step(self, action):
        # skipping
        if action == 0 : 
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message("action")
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message(str((self.skipTime+1)/self.fps))
            self.processList.append(self.frame)
            self.skipTime = 0
        if action == self.fps :
            self.skipTime += self.fps
        for a in range(1, self.fps):
            self.capIdx += 1
            ret, temp = self.cap.read()
            if not ret:
                self.cap.release()
                if self.isRun :
                    self.processedFrameList += self.processList
                    for frame in self.processedFrameList :
                        self.out.write(frame)
                    self.out.release()
                return -1, True
            self.frameList.append(temp)
            if a == action :
                self.processList.append(temp)
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("action")
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message(str((self.skipTime+action+1)/self.fps))
                self.skipTime = 0
            elif a > action : 
                self.processList.append(temp)
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("action")
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message(str((1)/self.fps))
        # new
        self.prev_frame = self.frameList[-1]
        ret, self.frame = self.cap.read()
        self.capIdx += 1
        if not ret:
            self.cap.release()
            if self.isRun :
                for frame in self.processedFrameList :
                    self.out.write(frame)
                self.out.release()
            return -1, True
        
        self.omnet.get_omnet_message()
        self.omnet.send_omnet_message("reward") 
        ratioA = float(self.omnet.get_omnet_message())
        self.omnet.send_omnet_message("ACK")
        
        newA = math.floor(ratioA*(self.fps))
        self._triggered_by_guide(newA, action)
        
        return self.state, False

    def _triggered_by_guide(self, newA, action):
        self.sendA = self.fps - action
        self.aSum += self.sendA
        self.ASum += self.targetA
        if self.isClusterexist :
            self.transList.append(self.fps-self.targetA)
            self.transList.append(self.state)
            self.transList.append(action)
        self.frameList = [self.frame]
        self.processList = [self.frame]
        self.prevA = self.targetA
        self.targetA = newA
        
        self.data.append(self.originState)
        # new state
        if self.isRun :
            self.processedFrameList += self.processList
            self.originState = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame)]
        else :
            self.originState = [get_MSE(self.prev_frame, self.frame), self.FFTList[self.curFrameIdx+self.fps]]
        if self.isClusterexist :
            self.state = cluster_pred(self.originState, self.model)
            self.transList.append(self.state)
            # reward
            if not self.isRun :
                self._get_reward()
            self.buffer.put(self.transList)
            # print(self.transList)
            self.transList = []
        return False
    
    def _get_FFT_List(self, exist=True):
        if not exist : 
            cap = cv2.VideoCapture(self.videoPath)
            self.FFTList = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                blur = get_FFT(frame)
                self.FFTList.append(blur)
            cap.release()
            # np.save(filePath, qTable)
            np.save("models/FFT_List_Jackson_1.npy", self.FFTList)
        else : 
            self.FFTList = np.load("models/FFT_List_Jackson_1.npy")
        return
    
    def _detect(self, exist=True):
        command = ["--weights", "models/yolov5s6.pt", "--source", self.videoPath, "--save-txt", "--save-conf", "--nosave"]
        if not exist : 
            inference(command) # cls, *xywh, conf
        fileList =  os.listdir(self.resultPath)
        self.objNumList = []  # 물체개수
        self._get_FFT_List(exist)
        for fileName in fileList :
            filePath = self.resultPath+"/"+fileName
            with open(filePath, 'r') as file :
                lines = file.readlines()
                self.objNumList.append(len(lines))
    
    def _get_reward(self):
        # get importance
        # normalizatino range
        rMin = 0
        rMax = 1
        rNewMin = -1
        rNewMax = 1
        for f in range(self.fps) :
            ww = self.w//2
            sIdx = max(0, f-ww)
            eIdx = min(self.fps-1, f+ww)
            sGap = f-sIdx
            eGap = eIdx-f
            if sGap < ww :
                eIdx += (ww-sGap)
            elif eGap < ww :
                sIdx -= (ww-eGap)
            eIdx = min(self.curFrameIdx+eIdx, len(self.objNumList))
            maxNum = max(self.objNumList[self.curFrameIdx+sIdx : eIdx+1])
            if self.curFrameIdx+f > len(self.objNumList) : break   
            importance = (self.objNumList[self.curFrameIdx + f] / maxNum) if maxNum != 0 else 0
            self.iList.append(importance)

        _, s, a, s_prime = self.transList
        origin_r = (sum(self.iList[self.curFrameIdx+a+1:]) - sum(self.iList[self.curFrameIdx:self.curFrameIdx+a+1]))
        r = (origin_r - rMin) * (rNewMax - rNewMin) / (rMax - rMin) + rNewMin
        self.transList.append(r)
        self.curFrameIdx += self.fps
        self.reward_sum += r
        
        if self.showLog :
            self.logList.append("A(t): "+str(self.prevA)+" action: "+str(a)+" A(t+1): "+str(self.targetA)+" reward: "+str(r))
        # print("===== reward =====")
        # print("r_blur:", r_blur)
        # print("r_dup:", r_dup)
        # print("r_net:", r_net)
        # print("R:", r)
        # print("==================")
        return

    def trans_show(self) :
        for row in self.logList :
            print(row)
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