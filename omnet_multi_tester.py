import numpy as np
import random
import argparse
import math
from tqdm import tqdm
from environment import Communicator
import threading

FPS = 30
ARRIVAL_MAX = 1.0
    
    
def parse_omnet_args() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--pipe_num", type=int, default=1)

    parser.add_argument("-s", "--step", type=int, default=150)
    parser.add_argument("-v", "--V", type=float, default=100)
    
    args = parser.parse_args()
    conf = vars(args)
    return conf

def send(omnet, time):
    omnet.get_omnet_message()
    omnet.send_omnet_message("action")
    omnet.get_omnet_message()
    omnet.send_omnet_message(str(time))
    return True


def receive(omnet):
    omnet.get_omnet_message()
    omnet.send_omnet_message("reward")
    path_cost = omnet.get_omnet_message()
    omnet.send_omnet_message("ACK")
    return float(path_cost)


def setting_omnet(pipe_num):
    if pipe_num == 1:
        omnet = Communicator("\\\\.\\pipe\\frame_drop_rl", 200000, debug_mode=False)
    else :
        omnet = Communicator("\\\\.\\pipe\\frame_drop_rl_"+str(pipe_num), 200000, debug_mode=False)
    return omnet


def testing(conf):
    omnet = setting_omnet(conf['pipe_num'])
    
    for epi in tqdm(range(conf['step'])):
        action = random.randint(0, FPS)
        for idx in range(FPS+1):
            if idx == action: send(omnet, (action+1)/FPS)
            elif idx > action: send(omnet, 1/FPS)

        path_cost = receive(omnet)
        arrival_rate = ARRIVAL_MAX if path_cost == 0 else min(ARRIVAL_MAX, conf['V'] / path_cost)
        arrival_frame_num = math.floor(arrival_rate*FPS)
        if debug_mode:
            print(f"action: {action}, path_cost: {path_cost}, max arrival rate with V: {arrival_rate}, max arrival frame num: {arrival_frame_num}")

    omnet.get_omnet_message()
    omnet.send_omnet_message("finish")
    
    
    debug_mode = True


if __name__ == "__main__":
    conf = parse_omnet_args()
    testing(conf)