"""
for offline-training
"""
from agent import Agent
from enviroment import FrameEnv

# hyperparameter -> change parameter using argparse


def _main():
    envV = FrameEnv("data/test.mp4")   # etc
    agentV = Agent()
    for epi in range(10000):           # request : how decide episode?
        done = False
        s = envV.reset()
        envV.omnet.init_pipe()         # request : add arg
        while not done:
            a = agentV.get_action(s)
            s, done = envV.step(a)
        if envV.buffer.size() > 300:
            for _ in range(50):
                trans = envV.buffer.get()
                agentV.Q_update(trans)
        agentV.decrease_eps()
    return


if __name__ == "__main__":
    # opt = _parge_opt()
    _main()
