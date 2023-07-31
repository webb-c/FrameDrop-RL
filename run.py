"""
for inference RL model
"""

import numpy as np
from agent import Agent
from enviroment import FrameEnv
import datetime
from torch.utils.tensorboard import SummaryWriter

def _get_q_table(filePath):
    qTable = np.load(filePath)
    return qTable

def _main(case) :
    # writer = SummaryWriter(logdir)
    videoPath = "data/RoadVideo-test.mp4"
    videoName = "RoadVideo-test"
    clusterPath = "models/cluster_RoadVideo-2.pkl"
    outVideoPath = "results/skiping-(caseS"+str(case)+").mp4"
    beta = 1.35
    fps = 30
    masking = False
    if case == 1 or case == 3 :
        masking = True
    if case == 1 :
        qTablePath = "models/q_table_mask__general.npy"
    elif case == 2 :
        qTablePath = "models/q_table_unmask__general.npy"
    elif case == 3 :
        qTablePath = "models/q_table_mask_YOLO_sim.npy"
    elif case == 4 :
        qTablePath = "models/q_table_unmask_YOLO_sim.npy"
    qTable = _get_q_table(qTablePath)
    envV = FrameEnv(videoName=videoName, videoPath=videoPath, clusterPath=clusterPath, fps=30, w=30, stateNum=15, resultPath="_", isClusterexist=True, isRun=True, beta=beta, runmode=case, masking=masking, outVideoPath=outVideoPath)   # etc
    agentV = Agent(qTable=qTable, eps_init=1.0, eps_decrese=0.01, eps_min=0.1, fps=30, lr=0.01, gamma=0.9, stateNum=15, isRun=True, masking=masking)
    done = False
    print("Ready ...")
    s = envV.reset()
    frame = 0
    while not done:
        print(frame)        
        if masking :
            requireskip = fps - envV.targetA
        else :
            requireskip = 0
        a = agentV.get_action(s, requireskip, False)
        s, done = envV.step(a)
        frame += 30
    envV.omnet.get_omnet_message()
    envV.omnet.send_omnet_message("finish")
    envV.omnet.close_pipe()
    print("a(t) :", envV.aSum, "A(t) :", envV.ASum)
    print("Test End!")

if __name__ == "__main__":
    # opt = _parge_opt()
    print("===========run.py==========")
    print("case 1) : node 6, layer 6(SLN), mask (using pipe 1)")
    print("case 2) : node 6, layer 6(SLN), unmask (using pipe 2)")
    print("case 3) : node 2, layer 5(YOLO), mask (using pipe 3)")
    print("case 4) : node 2, layer 5(YOLO), unmask (using pipe 4)")
    print("===========================")
    case = int(input("entering case : "))
    _main(case)