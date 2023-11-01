"""
for offline-training

Example of Usage :
    mask : 
    $ python train.py -qp models/q_table_mask -cp models/cluster_mask.pkl -m True
    
    unmask : 
    $ python train.py -qp models/q_table_unmask -cp models/cluster_unmask.pkl -m False
        
### for training Jetson continue... 
# mask :
# python train.py -qp models/q_table_mask_YOLO_jetson.npy  -con True -lr {learningrate} -ei {e init} -ed {e decrese} -em {em}
#
# unmask :
# python train.py -qp models/q_table_unmask_YOLO_jetson.npy  -con True -lr {learningrate}

"""

import numpy as np
from agent import Agent
from enviroment import FrameEnv
import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.get_state import cluster_train
from utils.yolov5.utils.general import print_args

def _save_q_table(qTable, filePath="models/q_table.npy") :
    print("save!")
    np.save(filePath, qTable)

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
    parser.add_argument("-ei", "--epsilonInit", type=int, default=1, help="epsilon init value")
    parser.add_argument("-ed", "--epsilonDecreseRate", type=float, default=0.005, help="epsilon decrese value")
    parser.add_argument("-em", "--epsilonMinimum", type=float, default=0.1, help="epsilon minimum value")
    
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-w", "--window", type=int, default=30, help="importance calculate object detect range")
    parser.add_argument("-s", "--stateNum", type=int, default=15, help="clustering state Number")
    parser.add_argument("-lr", "--lr", type=int, default=0.05, help="setting learning rate")
    parser.add_argument("-priorC", "--isClusterexist", type=str2bool, default=True, help="using pretrained cluster model?")
    
    parser.add_argument("-vp", "--videoPath", type=str, default="data/RoadVideo-train.mp4", help="training video path") # using Jetson-video : "data/jetson-train.mp4"
    # parser.add_argument("-vn", "--videoName", type=str, default="jetson-train", help="setting video name")

    parser.add_argument("-priorD", "--isDetectionexist", type=str2bool, default=True, help= "using predetected txt file?")
    parser.add_argument("-drp", "--detectResultPath", type=str, default="utils/yolov5/runs/detect/exp/labels", help="detect file path")
    # parser.add_argument("-cp", "--clusterPath", type=str, default="models/cluster_jetson-train.pkl", help="cluster model path")
    
    # *** soft ***
    parser.add_argument("-sw", "--softWeight", type=float, default=0.9, help="weight for combine soft and Q")
    parser.add_argument("-t", "--threshold", type=float, default=0.05, help="init threshold for select feasible set")
    parser.add_argument("-td", "--threshold_decrese", type=float, default=0.0002, help="threshold decrease rate")
    parser.add_argument("-std", "--std", type=float, default=5, help="std for gaussian distribution")
    parser.add_argument("-pt", "--pType", type=int, default=1, help="1 is combined 2 is Gaussian")
    # *** require ***
    parser.add_argument("-qp", "--qTablePath", type=str, default="models/q_table", help="qtable path")
    parser.add_argument("-b", "--beta", type=float, default=1.35, help="sensitive for number of objects")  # using Jetson-video : 0.5
    parser.add_argument("-m", "--masking", type=str2bool, default=True, help="using masking?")
    parser.add_argument("-soft", "--isSoft", type=str2bool, default=False, help="using soft-constraint?")
    parser.add_argument("-con", "--isContinue", type=str2bool, default=False, help="for Jetson Training")
    parser.add_argument("-pipe", "--pipeNum", type=int, default=1, help="pipe")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def _main(opt):
    # setting
    logdir="../total_logs/train/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # episode_maxlen = 300, 600, 2000
    episode_maxlen = 300
    epi_actions = 500

    data_maxlen = 10000
    replayBuffer_len = 1000
    replayBuffer_maxlen = 20000
    gamma = 0.9
    videoName = opt.videoPath.split("/")[-1].split(".")[0]
    clusterPath = "models/cluster_"+videoName+".pkl"
    
    # detect result 편의를 위해...
    if videoName == "jetson-train" :
        detectResultPath = "utils/yolov5/runs/detect/exp/labels"
    elif videoName == "RoadVideo-train" :
        detectResultPath = "utils/yolov5/runs/detect/exp2/labels"
    elif videoName == "jetson-train-new" :#TODO 
        detectResultPath = "utils/yolov5/runs/detect/exp3/labels"
    else :
        detectResultPath=opt.detectResultPath
        
    qTablePath = opt.qTablePath
    masking = opt.masking
    isClusterexist = opt.isClusterexist
    isContinue = opt.isContinue

    print_args(vars(opt))
    
    writer = SummaryWriter(logdir)
    envV = FrameEnv(videoName=videoName, videoPath=opt.videoPath, clusterPath=clusterPath, resultPath=detectResultPath, data_maxlen=data_maxlen, replayBuffer_maxlen=replayBuffer_maxlen, fps=opt.fps, w=opt.window, stateNum=opt.stateNum, isDetectionexist=opt.isDetectionexist, isClusterexist=isClusterexist, isRun=False, masking=masking, beta=opt.beta, runmode=opt.pipeNum, isSoft=opt.isSoft)   # etc
    agentV = Agent(eps_init=opt.epsilonInit, eps_decrese=opt.epsilonDecreseRate, eps_min=opt.epsilonMinimum, fps=opt.fps, lr=opt.lr, gamma=gamma, stateNum=opt.stateNum, threshold=opt.threshold, std=opt.std, pType=opt.pType, threshold_decrese=opt.threshold_decrese, isRun=False, masking=masking, isContinue=isContinue, isSoft=opt.isSoft)
    randAction = True
    for epi in range(episode_maxlen):       
        print("episode :", epi)
        done = False
        cluterVisualize = True
        showLog = False
        if epi == 0 or (epi % 50) == 0 or epi == episode_maxlen-1 :
            cluterVisualize = True
            showLog = True
        s = envV.reset(showLog=showLog)
        while not done:
            requireSkip = opt.fps - envV.targetA
            a = agentV.get_action(s, requireSkip, randAction)
            s, done = envV.step(a)
        if envV.buffer.size() > replayBuffer_len:
            randAction = False
            print("Q update ...")
            for _ in range(epi_actions):  # # of sampling count
                trans = envV.buffer.get()
                agentV.Q_update(trans)
            agentV.decrease_eps()
        # logging
        if isClusterexist :
            writer.add_scalar("Reward/one_Reward", envV.reward_sum, epi)
            writer.add_scalar("Network/Diff", (envV.ASum - envV.aSum), epi)
            writer.add_scalar("Network/target_A(t)", envV.ASum, epi)
            writer.add_scalar("Network/send_a(t)", envV.aSum, epi)
        if (epi % 50) == 0 or epi == episode_maxlen-1 :
            agentV.Q_show()
            qTable = agentV.get_q_table()
            if epi == episode_maxlen-1 :
                _save_q_table(qTable, qTablePath+".npy")
            # else :
            #    _save_q_table(qTable, qTablePath+"("+str(epi)+").npy")
        if showLog :
            envV.trans_show()
        # after first episode, make cluster model
        if not isClusterexist and epi == 1:
            
            isClusterexist = True
        print("buffer size: ", envV.buffer.size())
        envV.omnet.get_omnet_message()
        envV.omnet.send_omnet_message("finish")
        print("sum of A(t) : ", envV.ASum, "| sum of a(t) : ", envV.aSum)
    envV.omnet.close_pipe()
    return agentV.get_q_table()
    
if __name__ == "__main__":
    opt = parge_opt()
    _main(opt)
    print("Training Finish!")