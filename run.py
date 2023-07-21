"""
for inference RL model
"""

import numpy as np
from agent import Agent
from enviroment import FrameEnv
import datetime
from torch.utils.tensorboard import SummaryWriter

stateNum = 20
videoPath = "data/Jackson-2.mp4"
videoName = "/Jackson-2_"
resultPath = "utils/yolov5/runs/detect/exp4/labels"
logdir="../total_logs/test/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def _get_q_table(filePath="models/q_table.npy"):
    qTable = np.load(filePath)
    return qTable

def _main(qTable) :
    # TODO 프레임 실제로 구현하는 거, YOLO 분할하는거 추가...
    writer = SummaryWriter(logdir)
    envV = FrameEnv(videoName=videoName, videoPath=videoPath, resultPath=resultPath, isDetectionexist=False, isClusterexist=True, isRun=True)   # etc
    agentV = Agent(qTable=qTable, isRun=True)
    done = False
    s = envV.reset(isClusterexist=True)
    while not done:
        a = agentV.get_action(s)
        s, done = envV.step(a)
    print("Test End!")

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _get_q_table("models/q_table.npy")
    _main(qTable)