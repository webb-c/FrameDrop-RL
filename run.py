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
    videoPath = "data/jetson-test.mp4"
    videoName = "jetson-test"
    clusterPath = "models/cluster_jetson-train.pkl"
    outVideoPath = "results/skiping-(case"+str(case)+").mp4"
    beta = 0.5
    fps = 30
    masking = False
    if case == 1 or case == 2 or case == 3 or case == 5  :
        masking = True
    if case == 5 :
        qTablePath = "models/q_table_mask__general.npy"  # 도로
    elif case == 6 :
        qTablePath = "models/q_table_unmask__general.npy" # 도로
    elif case == 3 :
        qTablePath = "models/q_table_mask_YOLO_jetson_origin.npy" # 젯슨 촬영
    elif case == 4 : 
        qTablePath = "models/q_table_unmask_YOLO_jetson_origin.npy"  # 젯슨 촬영
    elif case == 1 :
        qTablePath = "models/q_table_mask_Agent1_origin.npy" # 젯슨 촬영
    elif case == 2 :
        qTablePath = "models/q_table_mask_Agent2_origin.npy" # 젯슨 촬영
    qTable = _get_q_table(qTablePath)
    envV = FrameEnv(videoName=videoName, videoPath=videoPath, clusterPath=clusterPath, fps=30, w=30, stateNum=15, resultPath="_", isClusterexist=True, isRun=True, beta=beta, runmode=case, masking=masking, outVideoPath=outVideoPath)   # etc
    agentV = Agent(qTable=qTable, eps_init=1.0, eps_decrese=0.01, eps_min=0.1, fps=30, lr=0.01, gamma=0.9, stateNum=15, isRun=True, masking=masking)
    done = False
    print("Ready ...")
    s = envV.reset()
    frame = 0
    aList = []
    uList = []
    AList = []
    while not done:
        print(frame)        
        if masking :
            requireskip = fps - envV.targetA
        else :
            requireskip = 0
        a = agentV.get_action(s, requireskip, False)
        AList.append(envV.targetA)
        uList.append(a)
        aList.append(30-a)
        s, done = envV.step(a)
        frame += 30
    envV.omnet.get_omnet_message()
    envV.omnet.send_omnet_message("finish")
    envV.omnet.close_pipe()
    print("a(t) :", envV.aSum, "A(t) :", envV.ASum)
    print("a(t) list :", aList)
    print("A(t) list :", AList)
    print("u(t) list :", uList)

if __name__ == "__main__":
    # opt = _parge_opt()
    print("===========run.py==========")
    print("case 1) : node 7, agent1, layer 6(SLN), mask (using pipe 1)")
    print("case 2) : node 7, agent2, layer 6(SLN), mask (using pipe 2)")
    print("case 3) : node 2, layer 5(YOLO), mask (using pipe 3)")
    print("case 4) : node 2, layer 5(YOLO), unmask (using pipe 4)")
    print("case 5) : node 6, layer 6(SLN), mask (using pipe 5)")
    print("case 6) : node 6, layer 6(SLN), unmask (using pipe 6)")
    print("===========================")
    case = int(input("entering case : "))
    _main(case)