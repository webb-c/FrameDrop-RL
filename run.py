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

import numpy as np
from pyprnt import prnt
import datetime
from typing import NameSpace, Tuple, Union, Dict
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from agent_ppo import PPOAgent
from environment import Environment
from utils.util import str2bool
import argparse


def parse_args() -> NameSpace: 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--video_path", type=str, default=None, help="testing video path")
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-model", "--model_path", type=str, default=None, help="trained model path")
    parser.add_argument("-use", "--useage", type=str2bool, default=False, help="using RL agent?")
    parser.add_argument("-out", "--output_path", type=str, default="./result/output.mp4", help="output video Path")
    
    return parser.parse_args()


def parse_name(conf:Dict[str, Union[str, int, bool, float]]) -> Dict[str, Union[str, int, bool, float]]:
    """모델 경로에 기록된 각종 정보를 통해 conf를 설정합니다.

    Args:
        conf (Dict[str, Union[str, int, bool, float]]): parse_args로 전달받은 기본 설정

    Returns:
        Dict[str, Union[str, int, bool, float]]: model_path parsing으로 분석한 설정이 추가된 dict
    """
    root_log = "./results/logs/test/"
    pass #TODO 
    return conf


def main(conf:Dict[str, Union[str, int, bool, float]]) :
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    conf = parse_name(conf)
    writer = SummaryWriter(conf['log_path'])
    
    prnt(conf)
    
    if not conf['usesage'] :
        pass #TODO 
    else :
        env = Environment(conf) 
        agent = Agent(conf, run=True)
        done = False
        print("Ready ...")
        s = env.reset()
        frame = 0
        aList = []
        uList = []
        AList = []
        
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
        
        print("a(t) :", env.sum_a, "A(t) :", env.sum_A)
        print("a(t) list :", aList)
        print("A(t) list :", AList)
        print("u(t) list :", uList)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
    print("Inference Finish!")