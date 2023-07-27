"""
for offline-training
"""
import numpy as np
from agent import Agent
from enviroment import FrameEnv
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.get_state import cluster_train
# hyperparameter -> change parameter using argparse
# for episode : frame count 9000
videoPath = "data/Jackson-1.mp4"
videoName = "/Jackson-1_"
resultPath = "utils/yolov5/runs/detect/exp2/labels"
qTablePath = "models/q_table_2"
clusterPath = "models/cluster_2.pkl"
logdir="../total_logs/train/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

fps = 30
w = 5

eps_init = 1
eps_decrese = 0.01
eps_min = 0.1

episoode_maxlen = 300
epi_actions = 500

stateNum = 15
data_len=500
data_maxlen=10000
replayBuffer_len=1000
replayBuffer_maxlen=20000

lr = 0.05     # learning rate
gamma = 0.9  # immediate and future
# V 1000000

def _main():
    isClusterexist = False
    isDetectionexist = True
    masking = True
    
    # isDetectionexist = False
    writer = SummaryWriter(logdir)
    envV = FrameEnv(videoName=videoName, videoPath=videoPath, clusterPath=clusterPath, resultPath=resultPath, data_maxlen=data_maxlen, replayBuffer_maxlen=replayBuffer_maxlen, fps=fps, w=w, stateNum=stateNum, isDetectionexist=isDetectionexist, isClusterexist=isClusterexist, isRun=False)   # etc
    agentV = Agent(eps_init=eps_init, eps_decrese=eps_decrese, eps_min=eps_min, fps=fps, lr=lr, gamma=gamma, stateNum=stateNum, isRun=False)
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
            requireSkip = fps - envV.targetA
            a = agentV.get_action(s, requireSkip, randAction)
            s, done = envV.step(a)
        if envV.buffer.size() > replayBuffer_len:
            if masking :
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
                envV.model = cluster_train(envV.model, np.array(envV.data),  clusterPath=clusterPath, visualize=cluterVisualize)
            elif (epi % 10) == 0 :
                envV.model = cluster_train(envV.model, np.array(envV.data),  clusterPath=clusterPath, visualize=cluterVisualize)
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

def _save_q_table(qTable, filePath="models/q_table.npy") :
    print("save!")
    np.save(filePath, qTable)

if __name__ == "__main__":
    # opt = _parge_opt()
    _main()
    print("Training Finish!")