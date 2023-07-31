"""
for inference RL model
"""

import numpy as np
from agent import Agent
from enviroment import FrameEnv
import datetime
from torch.utils.tensorboard import SummaryWriter

stateNum = 10
videoPath = "data/RoadVideo-2.mp4"
videoName = "/RoadVideo-2"
clusterPath = "models/cluster_mask__6.pkl"
qTablePath = "models/q_table_mask__6(299).npy"
outVideoPath = "results/RoadVideo-2-skip-mask__6.mp4"

def _get_q_table(filePath=qTablePath):
    qTable = np.load(filePath)
    return qTable

def _main(qTable) :
    # TODO 프레임 실제로 구현하는 거, YOLO 분할하는거 추가...
    # writer = SummaryWriter(logdir)
    episoode_maxlen = 300
    epi_actions = 500
    data_len = 500
    data_maxlen = 10000
    replayBuffer_len = 1000
    replayBuffer_maxlen = 20000
    gamma = 0.9
    envV = FrameEnv(videoName=videoName, videoPath=videoPath, clusterPath=clusterPath, resultPath="utils/yolov5/runs/detect/exp3/labels", data_maxlen=data_maxlen, replayBuffer_maxlen=replayBuffer_maxlen, fps=30, w=40, stateNum=15, isDetectionexist=True, isClusterexist=True, isRun=True, masking=True, beta=1.2)   # etc
    agentV = Agent(qTable=qTable, eps_init=1.0, eps_decrese=0.01, eps_min=0.1, fps=30, lr=0.01, gamma=0.9, stateNum=15, isRun=True, masking=True, isContinue=False)
    done = False
    print("Ready ...")
    s = envV.reset(isClusterexist=True)
    frame = 0
    while not done:
        print(frame)
        a = agentV.get_action(s, 0, False)
        s, done = envV.step(a)
        frame += 30
    print("Test End!")

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _get_q_table(qTablePath)
    _main(qTable)