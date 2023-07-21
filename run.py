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
outVideoPath = "results/Jackson-2-skip.mp4"
logdir="../total_logs/test/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def _get_q_table(filePath="models/q_table.npy"):
    qTable = np.load(filePath)
    return qTable

def _main(qTable) :
    # TODO 프레임 실제로 구현하는 거, YOLO 분할하는거 추가...
    writer = SummaryWriter(logdir)
    envV = FrameEnv(videoName=videoName, videoPath=videoPath, resultPath="./", isClusterexist=True, isRun=True, outVideoPath=outVideoPath)   # etc
    agentV = Agent(qTable=qTable, isRun=True)
    done = False
    print("Ready ...")
    s = envV.reset(isClusterexist=True)
    while not done:
        a = agentV.get_action(s)
        s, done = envV.step(a)
    print("Test End!")

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _get_q_table("models/q_table.npy")
    _main(qTable)