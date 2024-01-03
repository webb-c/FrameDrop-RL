"""
for inference RL model

Example of Usage :
    mask : 
    $ python run.py -qp models/q_table_mask -cp models/cluster_mask.pkl -m True
    
    unmask : 
    $ python run.py -qp models/q_table_unmask -cp models/cluster_unmask.pkl -m False
        

pre-trained models option :
    # [case1] video : data/RoadVideo-train.mp4, OMNeT++ env : 6-node topology, beta=1.35
    QTABLE
    - masked : models/q_table_mask__general.npy
    - unmasked : models/q_table_unmask__general.npy
    Cluster
    models/cluster_RoadVideo.pkl
    
    # [case2] video : data/jetson-train.mp4, OMNeT++ env : 2-node topology, beta=0.5
    QTABLE
    - masked : models/q_table_mask_YOLO_jetson.npy
    - unmasked : models/q_table_unmask_YOLO_jetson.npy 
    Cluster
    models/cluster_jetson=train.pkl
    
    # [case3] video : data/jetson-train.mp4, OMNeT++ env : 7-node topology, beta=0.5
    QTABLE
    - Agent1(masked) : models/q_table_mask_Agent1.npy
    - Agent2(unmasked) : models/q_table_mask_Agent2.npy
    Cluster
    models/cluster_jetson=train.pkl

"""

import os
import re
import numpy as np
from pyprnt import prnt
import datetime
from typing import Tuple, Union, Dict
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from agent_ppo import PPOAgent
from environment import Environment
from utils.parser import parse_test_args, parse_test_name, add_args
from utils.yolov5.detect import inference
from utils.cal_F1 import get_F1


def test(conf, start_time, writer):
    env = Environment(conf, run=True) 
    agent = Agent(conf, run=True)
    done = False
    print("Ready ...")
    s = env.reset()
    a_list, u_list, A_list = [], [], []
    
    step = 0
    while not done:       
        if conf['is_masking'] :
            require_skip = conf['fps'] - env.target_A
        else :
            require_skip = 0
        a = agent.get_action(s, require_skip, False)
        A_list.append(env.target_A)
        u_list.append(a)
        a_list.append(conf['fps']-a)
        writer.add_scalar("Network/Diff", (env.target_A - (conf['fps']-a)), step)
        writer.add_scalar("Network/target_A(t)", env.target_A, step)
        writer.add_scalar("Network/send_a(t)", conf['fps']-a, step)
        s, _, done = env.step(a)
        step += 1
    
    env.omnet.get_omnet_message()
    env.omnet.send_omnet_message("finish")
    env.omnet.close_pipe()
    
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    print("a(t) :", env.sum_a, "A(t) :", env.sum_A)
    if conf['log_network']:
        print("a(t) list :", a_list)
        print("A(t) list :", A_list)
        print("u(t) list :", u_list)


    print("Testing Finish with...")
    prnt(conf) 
    print("\n✱ start time :\t", start_time)
    print("✱ finish time:\t", finish_time)
    


def main(conf:Dict[str, Union[str, int, bool, float]]) -> bool:
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    conf, log_path = parse_test_name(conf, start_time)
    writer = SummaryWriter(log_path)
    prnt(conf)
    conf = add_args(conf)
    
    test(conf, start_time, writer)

    """
    if conf['f1_score'] :
        print("===== cheking F1 score =====")
        root_detection =  "./data/detect/test/"
        video_name = re.split(r"[/\\]", conf['video_path'])[-1].split(".")[0]
        model_name = re.split(r"[/\\]", conf['model_path'])[-1].split(".")[0]
        if not os.path.exists(conf['detection_path']): #TODO 
            command = ["--weights", "models/yolov5s6.pt", "--source", conf['video_path'], "--project", root_detection, "--name", video_name, "--save-txt", "--save-conf", "--nosave"]
            inference(command)
        
        command = ["--weights", "models/yolov5s6.pt", "--source", conf['video_path'], "--project", root_detection, "--name", video_name+"_"+model_name, "--save-txt", "--save-conf", "--nosave"]
        inference(command)
        
        origin_file = os.path.join(root_detection, video_name)
        skipped_file = os.path.join(root_detection, video_name+"_"+model_name)
        
        F1_score = get_F1(origin_file, skipped_file) #TODO 
        
        print("✲ F1 score: ")
    """
    return True


if __name__ == "__main__":
    opt = parse_test_args()
    conf = dict(**opt.__dict__)
    ret = main(conf)
    
    if not ret:
        print("Testing ended abnormally.")
