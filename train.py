"""
for offline-training

Example of Usage :
    mask : 
    $ python train.py -qp models/q_table_mask__1 -cp models/cluster_mask__1.pkl -m True
    
    unmask : 
    $ python train.py -qp models/q_table_2(unmask) -cp models/cluster_2(unmask).pkl -m False
        
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
    parser.add_argument("-vp", "--videoPath", type=str, default="data/Jackson-1.mp4", help="training video path")
    parser.add_argument("-vn", "--videoName", type=str, default="/Jackson-1_", help="setting video name")

    parser.add_argument("-priorD", "--isDetectionexist", type=str2bool, default=True, help="using predetected txt file?")
    parser.add_argument("-drp", "--detectResultPath", type=str, default="utils/yolov5/runs/detect/exp2/labels", help="detect file path")
    
    parser.add_argument("-ei", "--epsilonInit", type=int, default=1, help="epsilon init value")
    parser.add_argument("-ed", "--epsilonDecreseRate", type=float, default=0.01, help="epsilon decrese value")
    parser.add_argument("-em", "--epsilonMinimum", type=float, default=0.1, help="epsilon minimum value")
    
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-w", "--window", type=int, default=5, help="importance calculate object detect range")
    parser.add_argument("-s", "--stateNum", type=int, default=15, help="clustering state Number")
    parser.add_argument("-lr", "--lr", type=int, default=0.05, help="setting learning rate")
    parser.add_argument("-priorC", "--isClusterexist", type=str2bool, default=False, help="using pretrained cluster model?")
    
    # require
    parser.add_argument("-qp", "--qTablePath", type=str, default="models/q_table", help="qtable path")
    parser.add_argument("-cp", "--clusterPath", type=str, default="models/cluster.pkl", help="cluster model path")
    # *****
    parser.add_argument("-m", "--masking", type=str2bool, default=True, help="using masking?")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def _main(opt):
    # setting
    # V 1000000
    logdir="../total_logs/train/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    episoode_maxlen = 300
    epi_actions = 500
    data_len = 500
    data_maxlen = 10000
    replayBuffer_len = 1000
    replayBuffer_maxlen = 20000
    gamma = 0.9
    
    qTablePath = opt.qTablePath
    clusterPath = opt.clusterPath
    masking = opt.masking
    isClusterexist = opt.isClusterexist
    
    print_args(vars(opt))
    
    writer = SummaryWriter(logdir)
    envV = FrameEnv(videoName=opt.videoName, videoPath=opt.videoPath, clusterPath=clusterPath, resultPath=opt.detectResultPath, data_maxlen=data_maxlen, replayBuffer_maxlen=replayBuffer_maxlen, fps=opt.fps, w=opt.window, stateNum=opt.stateNum, isDetectionexist=opt.isDetectionexist, isClusterexist=isClusterexist, isRun=False, masking=masking)   # etc
    agentV = Agent(eps_init=opt.epsilonInit, eps_decrese=opt.epsilonDecreseRate, eps_min=opt.epsilonMinimum, fps=opt.fps, lr=opt.lr, gamma=gamma, stateNum=opt.stateNum, isRun=False, masking=masking)
    randAction = True
    for epi in range(episoode_maxlen):       
        print("episode :", epi)
        done = False
        cluterVisualize = False
        showLog = False
        if (epi % 50) == 0 or epi == episoode_maxlen-1 :
            cluterVisualize = True
            showLog = True
        s = envV.reset(isClusterexist=isClusterexist, showLog=showLog)
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
        # cluster update
        if len(envV.data) > data_len :
            if not isClusterexist :
                print("clustering ... ")
                envV.model = cluster_train(envV.model, np.array(envV.data), clusterPath=clusterPath, visualize=cluterVisualize, masking=masking)
            elif (epi % 10) == 0 :
                envV.model = cluster_train(envV.model, np.array(envV.data), clusterPath=clusterPath, visualize=cluterVisualize, masking=masking)
            isClusterexist = True
        if isClusterexist :
            writer.add_scalar("Reward/one_Reward", envV.reward_sum, epi)
            writer.add_scalar("Network/Diff", (envV.ASum - envV.aSum), epi)
            writer.add_scalar("Network/target_A(t)", envV.ASum, epi)
            writer.add_scalar("Network/send_a(t)", envV.aSum, epi)
        if (epi % 50) == 0 or epi == episoode_maxlen-1 :
            agentV.Q_show()
            qTable = agentV.get_q_table()
            _save_q_table(qTable, qTablePath+"("+str(epi)+").npy")
        if showLog :
            envV.trans_show()
        print("buffer size: ", envV.buffer.size())
        envV.omnet.get_omnet_message()
        envV.omnet.send_omnet_message("finish")
        isDetectionexist = True
        print("sum of A(t) : ", envV.ASum, "| sum of a(t) : ", envV.aSum)
    envV.omnet.close_pipe()
    return agentV.get_q_table()
    
if __name__ == "__main__":
    opt = parge_opt()
    _main(opt)
    print("Training Finish!")