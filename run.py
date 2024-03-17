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
from utils.util import save_parameters_to_csv


def test(conf, start_time, writer):
    if not conf['omnet_mode'] and conf['is_masking'] :
        assert True, "if you want masking mode, omnet mode must be set to True"

    env = Environment(conf, run=True)
    agent = Agent(conf, run=True)
    done = False
    print("Ready ...")
    s = env.reset()
    a_list, u_list, A_list = [], [], []
    
    step = 0
    while not done:
        # print(step, env.video_processor.idx)
        if conf['is_masking'] :
            require_skip = conf['fps'] - env.target_A
        else :
            require_skip = 0
        a = agent.get_action(s, require_skip, False)
        if conf['omnet_mode']:
            A_list.append(env.target_A)
        u_list.append(a)
        a_list.append(conf['fps']-a)
        if writer is not None:
            if conf['omnet_mode']:
                writer.add_scalar("Network/Diff", (env.target_A - (conf['fps']-a)), step)
                writer.add_scalar("Network/target_A(t)", env.target_A, step)
            writer.add_scalar("Network/send_a(t)", conf['fps']-a, step)
        s, _, done = env.step(a)
        step += 1
    
    if conf['omnet_mode']:
        env.omnet.get_omnet_message()
        env.omnet.send_omnet_message("finish")
        env.omnet.close_pipe()
    
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    if conf['log_network']:
        print("a(t) list :", a_list)
        if conf['omnet_mode']:
            print("A(t) list :", A_list)
        print("u(t) list :", u_list)
    
    fraction_value = env.video_processor.num_processed / env.video_processor.num_all 
    rounded_fraction = round(fraction_value, 4)
    
    if writer is not None:
        writer.add_scalar("Fractions", rounded_fraction, 1)
    
    A = 0
    if conf['omnet_mode']:
        A = env.sum_A
    return env.sum_a, A, finish_time, rounded_fraction, conf



def main(conf:Dict[str, Union[str, int, bool, float]]) -> bool:
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    conf = add_args(conf)
    conf, log_path = parse_test_name(conf, start_time)
    if not conf['debug_mode']:
        writer = SummaryWriter(log_path)
    else:
        writer = None
    prnt(conf)
    
    a, A, finish_time, rounded_fraction, conf = test(conf, start_time, writer)
    conf['fraction'] = rounded_fraction
    
    if conf['f1_score'] :
        root_detection =  "./data/detect/test/"
        video_name = re.split(r"[/\\]", conf['video_path'])[-1].split(".")[0]
        model_name = re.split(r"[/\\]", conf['model_path'])[-1].split(".")[0]
        
        origin_detection_path = os.path.join(root_detection, video_name + "/labels")
        if not os.path.exists(origin_detection_path):
            command = ["--weights", "models/yolov5s6.pt", "--source", conf['video_path'], "--project", root_detection, "--name", video_name, "--save-txt", "--save-conf", "--nosave"]
            inference(command)
        
        skip_detection_path = os.path.join(root_detection, conf['output_path'] + "/labels")
        if not os.path.exists(skip_detection_path):
            command = ["--weights", "models/yolov5s6.pt", "--source", conf['output_path'], "--project", root_detection, "--name", conf['output_path'], "--save-txt", "--save-conf", "--nosave"]
            inference(command)
        
        origin_list = os.listdir(origin_detection_path)
        skip_list = os.listdir(skip_detection_path)
        num_file = len(origin_list)
        total_F1 = 0
        for i in range(num_file):
            origin, skip = origin_list[i], skip_list[i]
            origin_file = os.path.join(origin_detection_path, origin)
            skip_file = os.path.join(skip_detection_path, skip)
            F1_score = get_F1(origin_file, skip_file)
            total_F1 += F1_score
        
        if writer is not None:
            writer.add_scalar("F1_score/total", total_F1, 1)
            writer.add_scalar("F1_score/average", total_F1/len(origin_list), 1)
        
        conf['f1_score'] = total_F1/len(origin_list)
    
    print("Testing Finish with...")
    prnt(conf)
    print("\n✱ start time :\t", start_time)
    print("✱ finish time :\t", finish_time)

    print("\n===== Networks =====")
    print("✲ Σa(t) :\t", a)
    print("✲ ΣA(t) :\t", A)
    
    print("\n===== Fractions =====")
    print("✲ processed frame rate :\t", rounded_fraction)

    if conf['f1_score'] :
        print("\n=====  F1 score =====")
        print("✲ total F1 score: ", total_F1)
        print("✲ average F1 score: ", total_F1/len(origin_list))
    
    if not conf['debug_mode']:
        save_parameters_to_csv(start_time, conf, train=False)
        print("\nsaving config is finished.")
    
    return True


if __name__ == "__main__":
    opt = parse_test_args()
    conf = dict(**opt.__dict__)
    ret = main(conf)
    
    if not ret:
        print("Testing ended abnormally.")
