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
    envV = FrameEnv("data/test.mp4", isClusterexist)   # etc
    agentV = Agent()
    for epi in range(10000):           # request : how decide episode?
        done = False
        s = envV.reset(isClusterexist)
        envV.omnet.init_pipe()         # request : add arg
        while not done:
            a = agentV.get_action(s)
            s, done = envV.step(a)
        if envV.buffer.size() > 300:
            for _ in range(50):
                trans = envV.buffer.get()
                agentV.Q_update(trans)
        # cluster update
        if len(envV.data) > 300 :
            envV.model = cluster_train(envV.model, envV.data)
            isClusterexist = True
        if isClusterexist :
            agentV.decrease_eps()
    return agentV.get_q_table()

def _save_q_table(qTable, filePath="models/q_table") :
    np.save(filePath, qTable)

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _main()
    _save_q_table(qTable)
    