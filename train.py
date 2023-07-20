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


def _main():
    stateNum = 20
    isClusterexist = False
    logdir="results/logs/train/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)
    envV = FrameEnv(videoPath="data/test.mp4", stateNum=stateNum, isClusterexist=isClusterexist)   # etc
    agentV = Agent(stateNum=stateNum)
    for epi in range(500):          
        print("episode :", epi)
        done = False
        s = envV.reset(isClusterexist=isClusterexist)
        while not done:
            a = agentV.get_action(s)
            s, done = envV.step(a)
        if envV.buffer.size() > 100:
            print("Q update ...")
            for _ in range(100):
                trans = envV.buffer.get()
                agentV.Q_update(trans)
        # cluster update
        if len(envV.data) > 100 :
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
        if epi % 50 :
            agentV.Q_show()
        envV.omnet.get_omnet_message()
        envV.omnet.send_omnet_message("finish") 
    return agentV.get_q_table()

def _save_q_table(qTable, filePath="models/q_table") :
    print("save!")
    np.save(filePath, qTable)

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _main()
    _save_q_table(qTable)
    