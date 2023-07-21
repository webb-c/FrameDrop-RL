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
stateNum = 20
videoPath = "data/Jackson-1.mp4"
videoName = "/Jackson-1_"
resultPath = "utils/yolov5/runs/detect/exp3/labels"
logdir="../total_logs/train/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_len=1000
data_maxlen=10000
replayBuffer_len=1000
replayBuffer_maxlen=10000
fps = 30
alpha = 0.5
beta = 2
w = 5
epi_actions = 1000
lr = 0.1
gamma = 0.9
episoode_maxlen = 500
eps_init = 1
eps_decrese = 0.01
eps_min = 0.1

def _main():
    isClusterexist = False
    isDetectionexist = True
    # isDetectionexist = False
    writer = SummaryWriter(logdir)
    envV = FrameEnv(videoName=videoName, videoPath=videoPath, resultPath=resultPath, data_maxlen=data_maxlen, replayBuffer_maxlen=replayBuffer_maxlen, fps=fps, alpha=alpha, beta=beta, w=w, stateNum=stateNum, isDetectionexist=isDetectionexist, isClusterexist=isClusterexist, isRun=False)   # etc
    agentV = Agent(eps_init=eps_init, eps_decrese=eps_decrese, eps_min=eps_min, fps=fps, lr=lr, gamma=gamma, stateNum=stateNum, isRun=False)
    #
    for epi in range(episoode_maxlen):          
        print("episode :", epi)
        done = False
        s = envV.reset(isClusterexist=isClusterexist)
        while not done:
            a = agentV.get_action(s)
            s, done = envV.step(a)
        if envV.buffer.size() > replayBuffer_len:
            print("Q update ...")
            for _ in range(epi_actions):  # # of sampling count
                trans = envV.buffer.get()
                agentV.Q_update(trans)
        # cluster update
        if len(envV.data) > data_len :
            if not isClusterexist :
                print("clustering ... ")
            envV.model = cluster_train(envV.model, np.array(envV.data), visualize=True)
            isClusterexist = True
        if isClusterexist :
            agentV.decrease_eps()
            writer.add_scalar("Reward/blur", envV.reward_sum[0], epi)
            writer.add_scalar("Reward/dup", envV.reward_sum[1], epi)
            writer.add_scalar("Reward/net", envV.reward_sum[2], epi)
            writer.add_scalar("Reward/total", envV.reward_sum[3], epi)
        if (epi % 50) == 0:
            agentV.Q_show()
        print("buffer size: ", envV.buffer.size())
        envV.omnet.get_omnet_message()
        envV.omnet.send_omnet_message("finish")
        isDetectionexist = True
    return agentV.get_q_table()

def _save_q_table(qTable, filePath="models/q_table.npy") :
    print("save!")
    np.save(filePath, qTable)

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _main()
    _save_q_table(qTable)
    