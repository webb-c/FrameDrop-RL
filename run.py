"""
for inference RL model

Example of Usage :
    mask : 
    $ python run.py -qp models/q_table_mask -cp models/cluster_mask.pkl -m True
    
    unmask : 
    $ python run.py -qp models/q_table_unmask -cp models/cluster_unmask.pkl -m False
        

pre-trained models option :
    # [case1] video : data/RoadVideo-train.mp4, OMNeT++ env : 6-node topology, beta=1.35
    QTABLE
    - masked : models/q_table_mask__general.npy
    - unmasked : models/q_table_unmask__general.npy
    Cluster
    models/cluster_RoadVideo.pkl
    
    # [case2] video : data/jetson-train.mp4, OMNeT++ env : 2-node topology, beta=0.5
    QTABLE
    - masked : models/q_table_mask_YOLO_jetson.npy
    - unmasked : models/q_table_unmask_YOLO_jetson.npy 
    Cluster
    models/cluster_jetson=train.pkl
    
    # [case3] video : data/jetson-train.mp4, OMNeT++ env : 7-node topology, beta=0.5
    QTABLE
    - Agent1(masked) : models/q_table_mask_Agent1.npy
    - Agent2(unmasked) : models/q_table_mask_Agent2.npy
    Cluster
    models/cluster_jetson=train.pkl

"""

import numpy as np
from agent import Agent
from enviroment import FrameEnv
import argparse

def _get_q_table(filePath):
    qTable = np.load(filePath)
    return qTable

def str2bool(v) :
    if isinstance(v, bool) :
        return v
    if v.lower() in ('true', 'yes', 't') :
        return True
    elif v.lower() in ('false', 'no', 'f') :
        return False
    else :
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parge_opt(known=False) :
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-w", "--window", type=int, default=30, help="importance calculate object detect range")
    parser.add_argument("-s", "--stateNum", type=int, default=15, help="clustering state Number")
    parser.add_argument("-priorC", "--isClusterexist", type=str2bool, default=True, help="using pretrained cluster model?")
    
    parser.add_argument("-vp", "--videoPath", type=str, default="data/jetson-test.mp4", help="training video path")
    parser.add_argument("-vn", "--videoName", type=str, default="jetson-test", help="setting video name")

    parser.add_argument("-cp", "--clusterPath", type=str, default="models/cluster_jetson-train.pkl", help="cluster model path")
    parser.add_argument("-con", "--isContinue", type=str2bool, default=False, help="for Jetson Training")
    parser.add_argument("-sh", "--showflowrate", type=str2bool, default=True, help="show action and flow rate")
    
    # *** require ***
    parser.add_argument("-qp", "--qTablePath", type=str, default="models/q_table", help="trinaed qtable path")
    parser.add_argument("-b", "--beta", type=float, default=0.5, help="sensitive for number of objects")
    parser.add_argument("-m", "--masking", type=str2bool, default=True, help="using masking?")
    parser.add_argument("-pipe", "--pipeNum", type=int, default=1, help="using pipe number")
    parser.add_argument("-op", "--outVideoPath", type=str, default="results/skiping.mp4", help="output video Path")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

def _main(opt) :
    qTable = _get_q_table(opt.qTablePath)
    envV = FrameEnv(videoName=opt.videoName, videoPath=opt.videoPath, clusterPath=opt.clusterPath, fps=opt.fps, w=opt.window, stateNum=opt.stateNum, resultPath="-", isClusterexist=opt.isClusterexist, isRun=True, beta=opt.beta, runmode=opt.pipeNum, masking=opt.masking, outVideoPath=opt.outVideoPath)   # etc
    agentV = Agent(qTable=qTable, eps_init=1.0, eps_decrese=0.01, eps_min=0.1, fps=opt.fps, lr=0.15, gamma=0.9, stateNum=opt.stateNum, isRun=True, masking=opt.masking)
    done = False
    print("Ready ...")
    s = envV.reset()
    frame = 0
    aList = []
    uList = []
    AList = []
    while not done:
        print(frame)        
        if opt.masking :
            requireskip = opt.fps - envV.targetA
        else :
            requireskip = 0
        a = agentV.get_action(s, requireskip, False)
        AList.append(envV.targetA)
        uList.append(a)
        aList.append(opt.fps-a)
        s, done = envV.step(a)
        frame += opt.fps
    envV.omnet.get_omnet_message()
    envV.omnet.send_omnet_message("finish")
    envV.omnet.close_pipe()
    if opt.showflowrate :
        print("a(t) :", envV.aSum, "A(t) :", envV.ASum)
        print("a(t) list :", aList)
        print("A(t) list :", AList)
        print("u(t) list :", uList)

if __name__ == "__main__":
    opt = parge_opt()
    _main(opt)
    print("Inference Finish!")