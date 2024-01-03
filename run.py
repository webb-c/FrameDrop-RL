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
from typing import NameSpace, Tuple, Union, Dict
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from agent_ppo import PPOAgent
from environment import Environment
from utils.util import str2bool
from utils.yolov5.detect import inference
from utils.cal_F1 import get_F1
import argparse


def parse_args() -> NameSpace: 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--video_path", type=str, default=None, help="testing video path")
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-model", "--model_path", type=str, default=None, help="trained model path")
    parser.add_argument("-mask", "--masking", type=str2bool, default=False, help="using lyapunov based guide?")
    parser.add_argument("-out", "--output_path", type=str, default="./result/output.mp4", help="output video Path")
    parser.add_argument("-f1", "--f1_score", type=str2bool, default=True, help="showing f1 score")
    
    return parser.parse_args()


def parse_name(conf:Dict[str, Union[str, int, bool, float]], start_time:str) -> Dict[str, Union[str, int, bool, float]]:
    """모델 경로에 기록된 각종 정보를 통해 conf를 설정합니다.

    Args:
        conf (Dict[str, Union[str, int, bool, float]]): parse_args로 전달받은 기본 설정

    Returns:
        Dict[str, Union[str, int, bool, float]]: model_path parsing으로 분석한 설정이 추가된 dict
    """
    root_log = "./results/logs/test/"
    pass #TODO 
    return conf


def test(conf):
    env = Environment(conf) 
    agent = Agent(conf, run=True)
    done = False
    print("Ready ...")
    s = env.reset()
    frame = 0
    aList = []
    uList = []
    AList = []
    
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    while not done:
        print(frame)        
        if opt.masking :
            requireskip = opt.fps - env.target_A
        else :
            requireskip = 0
        a = agent.get_action(s, requireskip, False)
        AList.append(env.target_A)
        uList.append(a)
        aList.append(opt.fps-a)
        s, done = env.step(a)
        frame += opt.fps
    
    env.omnet.get_omnet_message()
    env.omnet.send_omnet_message("finish")
    env.omnet.close_pipe()
    
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    print("a(t) :", env.sum_a, "A(t) :", env.sum_A)
    print("a(t) list :", aList)
    print("A(t) list :", AList)
    print("u(t) list :", uList)
    
    print("Testing Finish with...")
    prnt(conf)
    print("\n✱ start time :\t", start_time)
    print("✱ finish time:\t", finish_time)
    


def main(conf:Dict[str, Union[str, int, bool, float]]) :
    conf = parse_name(conf)
    writer = SummaryWriter(conf['log_path'])
    prnt(conf)
    
    test(conf)

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
    # drop한 프레임도 파일 생기게 그냥 0으로 된 프레임 만들어서 줘야될듯?


if __name__ == "__main__":
    opt = parse_args()
    conf = dict(**opt.__dict__)
    ret = main(conf)
    
    if not ret:
        print("Testing ended abnormally.")
