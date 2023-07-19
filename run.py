"""
for inference RL model
"""
import numpy as np
from agent import Agent
from enviroment import FrameEnv

def _save_q_table(filePath="models/q_table.npy"):
    qTable = np.load(filePath)
    return qTable

def _main(qTable) :
    envV = FrameEnv("data/test.mp4", isClusterexist=True)
    agentV = Agent(qTable)
    # request : test code

if __name__ == "__main__":
    # opt = _parge_opt()
    qTable = _save_q_table()
    _main(qTable)