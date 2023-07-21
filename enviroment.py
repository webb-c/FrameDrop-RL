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

    def put(self, transList):
        for trans in transList:
            self.buffer.append(trans)
            
    def get(self):
        trans = random.choice(self.buffer)  # batch size = 1
        return trans

    def size(self):
        return len(self.buffer)


class FrameEnv():
    def __init__(self, videoName, videoPath, resultPath, data_maxlen=10000, replayBuffer_maxlen=10000, fps=30, alpha=0.5, beta=2, w=5, stateNum=20, isDetectionexist=True, isClusterexist=False, isRun=False, outVideoPath="./output.mp4"):
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
        # hyper-parameter
        self.alpha = alpha 
        self.beta = beta
        self.w = w
        self.model = cluster_init(k=stateNum)
        if self.isClusterexist :  
            self.model = cluster_load()
        if not self.isRun :
            self._detect(self.isDetectionexist)
        # state
        self.reset()
        if self.isRun :
            self.processedFrameList = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.outVideoPath = outVideoPath
            self.out = cv2.VideoWriter(outVideoPath, fourcc, self.fps, (self.frame.shape[1], self.frame.shape[0]))
            
    def reset(self, isClusterexist=False):
        self.reward_sum = [0, 0, 0, 0] # r_dup, r_blur, r_net, r_total
        self.iList = []
        self.isClusterexist = isClusterexist
        self.cap = cv2.VideoCapture(self.videoPath)
        self.idx = 0
        self.curFrameIdx = 1
        self.prevA = self.fps
        self.targetA = self.fps
        _, f1 = self.cap.read()
        _, f2 = self.cap.read()
        self.frameList = [f1, f2]
        self.processList = [f2]
        self.prev_frame = self.frameList[-2]
        self.frame = self.frameList[-1]
        self.net = self._get_sNet()
        if self.isRun :
            self.originState = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net]
        else :
            self.originState = [get_MSE(self.prev_frame, self.frame), self.FFTList[self.curFrameIdx], self.net]
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
                self.cap.release()
                self.processedFrameList += self.processList
                for frame in self.processedFrameList :
                    self.out.write(frame)
                self.out.release()
                return -1, True
            if len(self.frameList) >= self.fps:
                # call OMNeT++
                guided = True
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("reward") 
                ratioA = float(self.omnet.get_omnet_message())
                self.omnet.send_omnet_message("ACK")
                newA = math.floor(ratioA*(self.fps))
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
                self.transList.append([self.state, action])
            # print("state: ",self.originState,"action: ", action)
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message("action")
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message(str((action+1)/self.fps))
            self.data.append(self.originState)
            self.curFrameIdx += (action+1)
        
        self.net = self._get_sNet()
        if self.isRun :
            self.originState = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net]
        else :
            self.originState = [get_MSE(self.prev_frame, self.frame), self.FFTList[self.curFrameIdx], self.net]
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
            self.transList.append([self.state, a])
        # print("state: ",self.originState,"action: ", a)
        self.omnet.get_omnet_message()
        self.omnet.send_omnet_message("action")
        self.omnet.get_omnet_message()
        self.omnet.send_omnet_message(str((a+1)/self.fps))
        self.data.append(self.originState)
        # new state (curr_trans s_prime)
        # print("new A :", newA)
        self.prev_frame = self.frameList[-1]
        self.frame = temp
        self.frameList = [self.frame]
        self.processedFrameList += self.processList
        self.processList = [self.frame]
        self.prevA = self.targetA
        self.targetA = newA
        self.net = self._get_sNet()
        self.curFrameIdx += (a+1)
        if self.isRun :
            self.originState = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame), self.net]
        else :
            self.originState = [get_MSE(self.prev_frame, self.frame), self.FFTList[self.curFrameIdx], self.net]
        if self.isClusterexist :
            self.state = cluster_pred(self.originState, self.model)
            # curr_trans append s_prime
            self.transList[-1].append(self.state)
            # reward
            if not self.isRun :
                self._get_reward()
            self.buffer.put(self.transList[:])
            self.transList = []
        # subTask 2
        na = action-a-1
        if na >= 0:
            # curr_trans (s, a)
            if self.isClusterexist : 
                self.transList = [[self.state, na]]
            # print("state: ",self.originState,"action: ", na)
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message("action")
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message(str((na+1)/self.fps))
            # ret = self.omnet.get_omnet_message()
            # if ret != "ACK" :
            #     print("error in action")
            #     return
            self.curFrameIdx += (na+1)
            return False
        return True
    
    def _get_sNet(self):
        return (self.targetA - len(self.processList))/(self.fps + 1 - len(self.frameList))
    
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
            np.save("models/FFT_List.npy", self.FFTList)
        else : 
            self.FFTList = np.load("models/FFT_List.npy")
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
        length = len(self.transList)
        # get importance
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
            eIdx = min(self.idx+eIdx, len(self.objNumList))
            maxNum = max(self.objNumList[self.idx+sIdx : eIdx+1])
            if self.idx+f > len(self.objNumList) : break   
            inv_importance = 10*(1 - self.objNumList[self.idx + f] / maxNum) if maxNum != 0 else 0
            self.iList.append(inv_importance)
        A_diff = self.targetA - self.prevA
        for i in range(length):
            s, a, s_prime = self.transList[i]
            # request addition (YOLO -> self.frameList detect)
            r_blur = (sum(self.iList[self.idx:self.idx+a+1]) / a) if a != 0 else 0
            self.F1List = []
            refFrame = self.resultPath+self.videoName+str(self.idx+1)+".txt"
            for k in range(1, a+1) :
                skipFrame = self.resultPath+self.videoName+str(self.idx+k+1)+".txt"
                self.F1List.append(1 - get_F1(refFrame, skipFrame))
            r_dup = sum(self.F1List)
            # r_net
            if A_diff >= 0 :
                r_net = (a/self.fps)*(A_diff) + self.targetA
            else :
                r_net = self.beta * (1-a/self.fps)*(A_diff) + self.targetA
            r_net = r_net / 10
            r = (1 - self.alpha) * (r_blur + r_dup) + self.alpha * (r_net)
            self.transList[i].append(r)
            self.idx += a+1
            self.reward_sum[0] += r_blur
            self.reward_sum[1] += r_dup
            self.reward_sum[2] += r_net
            self.reward_sum[3] += r
            # print("===== reward =====")
            # print("r_blur:", r_blur)
            # print("r_dup:", r_dup)
            # print("r_net:", r_net)
            # print("R:", r)
            # print("==================")
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