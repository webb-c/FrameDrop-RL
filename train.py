"""
for offline-training
"""
import numpy as np
from agent import Agent
from enviroment import FrameEnv
from utils.get_state import cluster_train
# hyperparameter -> change parameter using argparse


def _main():
    isClusterexist = False
    envV = FrameEnv("data/test.mp4", isClusterexist=isClusterexist)   # etc
    agentV = Agent()
    for epi in range(10000):           # request : how decide episode?
        print("episode :", epi)
        done = False
        s = envV.reset(isClusterexist)
        print(envV.isClusterexist)
        while not done:
            a = agentV.get_action(s)
            s, done = envV.step(a)
        print("buffer size:", envV.buffer.size())
        if envV.buffer.size() > 50:
            print("Q update ...")
            for _ in range(50):
                trans = envV.buffer.get()
                agentV.Q_update(trans)
        # cluster update
        if len(envV.data) > 50 :
            print("clustering ... ")
            envV.model = cluster_train(envV.model, np.array(envV.data), visualize=True)
            isClusterexist = True
        if isClusterexist :
            agentV.decrease_eps()
        envV.omnet.get_omnet_message()
        envV.omnet.send_omnet_message("finish") 
    return agentV.get_q_table()

def _save_q_table(qTable, filePath="models/q_table") :
    np.save(filePath, qTable)

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _main()
    _save_q_table(qTable)
    