"""
for offline-training

Example of Usage :
    mask : 
    $ python train.py -qp models/q_table_mask -cp models/cluster_mask.pkl -m True
    
    unmask : 
    $ python train.py -qp models/q_table_unmask -cp models/cluster_unmask.pkl -m False
        
### for training Jetson continue... 
# mask :
# python train.py -qp models/q_table_mask_YOLO_jetson.npy  -con True -lr {learningrate} -ei {e init} -ed {e decrese} -em {em}
#
# unmask :
# python train.py -qp models/q_table_unmask_YOLO_jetson.npy  -con True -lr {learningrate}

"""
import msvcrt
import sys
import signal
import os
import cv2
import re
from pyprnt import prnt
from typing import Tuple, List, Dict, Union
import numpy as np
from tqdm import tqdm
from agent import Agent
from agent_ppo import PPOAgent
from environment import Environment
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
from utils.get_state import cluster_train, cluster_init
from utils.cal_quality import get_FFT, get_MSE
from utils.parser import parse_train_args
from utils.util import save_parameters_to_csv

from utils.yolov5.detect import inference


def path_manager(video_path: str, state_num: int, radius: int, state_method: int, action_dim: int) -> Tuple[str, str, str] :
    """ video path를 전달받아서 cluster, detection, FFT 경로를 지정하여 반환합니다.

    Args:
        video_path (str): 학습에 사용할 영상의 경로
        state_num (int): 학습에 사용하는 cluster의 개수
        
    Returns:
        Tuple[cluster_path, detection_path, FFT_path]
    """
    root_cluster, root_detection, root_FFT = "./models/cluster/", "./data/detect/train/", "./data/FFT/"
    video_name = re.split(r"[/\\]", video_path)[-1].split(".")[0]
    cluster_video_name = video_name.replace("_", "")
    if state_method == 0:
        cluster_path = os.path.join(root_cluster, cluster_video_name + "_" + str(state_num) + "_" + str(radius) + "_" + str(state_method)+ ".pkl")
    elif state_method == 1:
        cluster_path = os.path.join(root_cluster, cluster_video_name + "_" + str(state_num) + "_" + str(radius) + "_" + str(action_dim) + "_" + str(state_method) + ".pkl")
    elif state_method == 2:
        cluster_path = os.path.join(root_cluster, cluster_video_name + "_" + str(state_num) + "_" + str(action_dim) + "_" + str(state_method) + ".pkl")
    
    detection_path = os.path.join(root_detection, video_name + "/labels")
    FFT_path = os.path.join(root_FFT, video_name + "_" + str(radius) + ".npy")
    
    return cluster_path, detection_path, FFT_path


def logging_mannager(start_time: str, conf: Dict[str, Union[str, bool, int, float]], default_conf: Dict[str, Union[str, bool, int, float]]) -> Tuple[str, str]:
    """학습관련 argument가 기입된 conf를 전달받아 로깅, 지정에 사용할 경로를 반환합니다.
    Args:
        start_time: 학습을 시작한 시간
        conf (Dict): train argument
        default_conf (Dict): train argument's default value
        
    Returns:
        Tuple[log_path, save_path]
    """
    root_log = "./results/logs/train/"
    root_save = "./models/"
    skip_list = ['learn_method', 'pipe_num', 'fps', 'episode_num', 'is_masking', 'V', 'debug_mode']
    name = start_time
    if default_conf is None:
        for arg, value in conf.items():
            name += f"_{arg}_{value}"
    else:
        for arg, value in conf.items():
            default_value = default_conf.get(arg)
            if arg in skip_list:
                continue
            if value != default_value:
                if arg == 'video_path':
                    value = re.split(r"[/\\]", value)[-1].split(".")[0]
                arg = arg.replace("_", "")
                if isinstance(value, str):
                    value = value.replace("_", "")
                name += f"_{arg}_{value}"
    
    log_path = os.path.join(root_log, name)
    root_save = os.path.join(root_save, "ndarray")
    save_path = os.path.join(root_save, name + ".npy")
    
    return log_path, save_path, name


def save_name_mannager(epi, name):
    """현재까지 학습한 episode를 전달하면 .npy 파일을 저장하기 위해 사용할 경로를 반환합니다.

    Args:  
        epi:
    """
    root_save = "./models/"
    name = name+"_e_"+str(epi)
    root_save = os.path.join(root_save, "ndarray")
    save_path = os.path.join(root_save, name + ".npy")
    
    return save_path
    

def verifier(conf: Dict[str, Union[str, bool, int, float]], cluster_path: str, detection_path: str, FFT_path: str):
    """경로를 전달받아서 데이터가 존재하는지 검증하고 없다면 만들어냅니다.

    Args:
        conf (Dict[str, Union[bool, int, float]]): train argument
        cluster_path (str): _description_
        detection_path (str): _description_
        FFT_path (str): _description_
    """
    
    if not os.path.exists(detection_path):
        print("start making detection files ...")
        root_detection =  "./data/detect/train/"
        video_name = re.split(r"[/\\]", conf['video_path'])[-1].split(".")[0]
        command = ["--weights", "models/yolov5s6.pt", "--source", conf['video_path'], "--project", root_detection, "--name", video_name, "--save-txt", "--save-conf", "--nosave"]
        inference(command)
        print("finish making detection files !\n")
    
    if not os.path.exists(FFT_path):
        print("start making blur level file ...")
        cap = cv2.VideoCapture(conf['video_path'])
        FFTList = []
        idx = 0
        r = conf["radius"]
        while True:
            print("video 1/1", idx)
            ret, frame = cap.read()
            if not ret:
                break
            blur = get_FFT(frame, radius=r)
            FFTList.append(blur)
            idx += 1
        cap.release()
        np.save(FFT_path, FFTList)
        print("finish making blur level file !\n")
    
    if not os.path.exists(cluster_path):
        print("start making cluster model ...")
        cluster = cluster_init(state_num=conf['state_num'])
        cap = cv2.VideoCapture(conf['video_path'])
        
        if conf['state_method'] == 0:
            FFTList = np.load(FFT_path)
            data = []
            _, f_prev = cap.read()
            idx = 0
            data.append([0, FFTList[idx]])
            while True :
                idx += 1
                ret, f_cur = cap.read()
                if not ret :
                    cap.release()
                    break
                data.append([get_MSE(f_prev, f_cur), FFTList[idx]])
                f_prev = f_cur
        
        
        elif conf['state_method'] == 1:
            FFTList = np.load(FFT_path)
            frame_count = 0
            frame_list = []
            data = []
            idx = 0
            while True :
                ret, frame = cap.read()
                if not ret :
                    cap.release()
                    break
                
                frame_count += 1
                frame_list.append(frame)
                if len(frame_list) > conf['action_dim']:    
                    for k in range(1, conf['action_dim']+1):
                        data.append([get_MSE(frame_list[0], frame_list[k]), FFTList[idx+k]])
                    frame_list.pop(0)
                idx += conf["action_dim"]

        
        elif conf['state_method'] == 2:
            frame_count = 0
            frame_list = []
            data = []
            idx = 0
            while True :
                ret, frame = cap.read()
                if not ret :
                    cap.release()
                    break
                
                frame_count += 1
                frame_list.append(frame)
                if len(frame_list) > conf['action_dim']:    
                    for k in range(1, conf['action_dim']+1):
                        data.append([get_MSE(frame_list[0], frame_list[k]), get_MSE(frame_list[k-1], frame_list[k])])
                    frame_list.pop(0)
                idx += conf["action_dim"]
        
        cluster = cluster_train(cluster, np.array(data), cluster_path)
        print("finish making cluster model !\n")


def input_with_timeout(timeout):
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if msvcrt.kbhit():
            return sys.stdin.readline().rstrip()
    raise TimeoutError()


def main(conf: Dict[str, Union[str, bool, int, float]], default_conf: Dict[str, Union[str, bool, int, float]]) -> bool :
    """argument를 전달받아, 그 설정대로 강화학습을 수행합니다.

    Args:
        conf (Dict[str, Union[bool, int, float]]): train argument
        default_conf (Dict[str, Union[bool, int, float]]): train argument's default value

    Returns:
        bool: 정상적으로 학습이 종료되면 True를 반환합니다.
    """
    if not conf['omnet_mode'] and conf['is_masking'] :
        assert True, "if you want masking mode, omnet mode must be set to True"
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S") 
    cluster_path, detection_path, FFT_path = path_manager(conf['video_path'], conf['state_num'], conf["radius"], conf["state_method"], conf["action_dim"])
    verifier(conf, cluster_path, detection_path, FFT_path)
    log_path, save_path, name = logging_mannager(start_time, conf, default_conf)
    if not conf['debug_mode'] :
        writer = SummaryWriter(log_path)
    conf['cluster_path'] = cluster_path
    conf['detection_path'] = detection_path
    conf['FFT_path'] = FFT_path
    prnt(conf)

    if conf['learn_method'] == "Q" :
        env = Environment(conf)
        agent = Agent(conf, run=False)
        rand = True

        #! Handler 등록
        def handler(signum, frame):
            print('Ctrl+C가 눌렸습니다. 현재 학습중인 Agent를 저장하고 종료합니다.')
            path = save_name_mannager(epi, name)
            agent.save_model(path)
            save_parameters_to_csv(start_time, conf, train=True)
            print("저장 완료!\n")
            sys.exit(0)
        
        if not conf['debug_mode']:
            signal.signal(signal.SIGINT, handler)
    
        print("train start !")
        for epi in tqdm(range(conf['episode_num'])):
            done = False
            show_log = False
            if epi == 0 or (epi % 50) == 0 or epi == conf['episode_num']-1 :
                show_log = True
            s = env.reset(show_log=show_log)
            # start_time = time.time()     
            while not done:
                if conf['omnet_mode']:
                    require_skip = conf['action_dim'] - env.target_A
                else :
                    require_skip = 0
                a = agent.get_action(s, require_skip, rand)
                s, _, done = env.step(a)
            
            if env.buffer.get_size() < conf['start_buffer_size']:
                print("buffer size: ", env.buffer.get_size())
            else:
                rand = False
                prev_qtable = agent.get_qtable()
                for _ in range(conf["sampling_num"]):  # # of sampling count
                    trans = env.buffer.get_data()
                    agent.update_qtable(trans)
                agent.decrease_eps()
                
                if agent.is_converge(prev_qtable):
                    print("early stop in episode", epi)
                    break
            
            # logging
            if not conf["debug_mode"]:
                writer.add_scalar("Reward/all", env.reward_sum, epi)
                writer.add_scalar("Reward/"+conf['reward_method'], env.reward_sum, epi)
                writer.add_scalar("Network/send_a(t)", env.sum_a, epi)
                if conf['omnet_mode']:
                    writer.add_scalar("Network/Diff", (env.sum_A - env.sum_a), epi)
                    writer.add_scalar("Network/target_A(t)", env.sum_A, epi)
                    print("sum of A(t) : ", env.sum_A, "| sum of a(t) : ", env.sum_a)
            
            if epi == 0 or (epi % 50) == 0:
                agent.show_qtable()
                env.show_trans()
                if not conf["debug_mode"]:
                    path = save_name_mannager(epi, name)
                    agent.save_model(path)
            
            if conf['omnet_mode']:
                env.omnet.get_omnet_message()
                env.omnet.send_omnet_message("finish")
            
            # end_time  = time.time()
            # print("time: {:.2f}s".format(end_time - start_time))
            
    
    #TODO:
    elif conf['learn_method'] == "PPO" :
        agent = PPOAgent(conf)
        total_reward = 0
        print_interval = 50
        for episode in tqdm(range(conf['episode_num'])):
            epi_reward = 0
            state = env.reset(show_log=True)
            done = False
            while not done :
                for t in range(conf['rollout_len']):
                    guide = conf['fps'] - env.target_A
                    action, action_prob = agent.get_actions(state, guide)
                    # print(action)
                    action = action.item()
                    state_prime, reward, done = env.step(action)
                    if done : 
                        break
                    agent.put_data((state, action, reward, state_prime, action_prob, done, guide)) 
                    epi_reward += reward
                    state = state_prime
                loss, value_loss, policy_loss = agent.train_net()
            total_reward += epi_reward
            
            # record total_reward & avg_reward & loss for each episode
            print("sum of A(t) : ", env.sum_A, "| sum of a(t) : ", env.sum_a)
            if not conf["debug_mode"]:
                writer.add_scalar("Reward/"+conf['reward_method'], epi_reward, episode)
                writer.add_scalar("Network/Diff", (env.sum_A - env.sum_a), episode)
                writer.add_scalar("Network/target_A(t)", env.sum_A, episode)
                writer.add_scalar("Network/send_a(t)", env.sum_a, episode)
                if loss is not None :
                    writer.add_scalar("loss", loss.mean().item(), episode)
                    writer.add_scalar("value_loss", sum(value_loss).mean().item(), episode)
                    writer.add_scalar("policy_loss", sum(policy_loss).mean().item(), episode)

            if episode % print_interval == 0 :
                print("\n# of episode :{}, avg reward : {:.2f}, total reward : {:.2f}".format(episode, total_reward/print_interval, total_reward))
                total_reward = 0

            env.omnet.get_omnet_message()
            env.omnet.send_omnet_message("finish")
    
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    if conf['omnet_mode']: 
        env.omnet.close_pipe()
    
    ret = agent.save_model(save_path)
    if not ret:
        print("Saving model ended abnormally")
    
    print("Training Finish with...")
    prnt(conf)
    
    print("\n✱ start time :\t", start_time)
    print("✱ finish time:\t", finish_time)
    
    if not conf['debug_mode']:
        save_parameters_to_csv(start_time, conf, train=True)
    
    return True


if __name__ == "__main__":
    conf, default_conf = parse_train_args()
    ret = main(conf, default_conf)
    
    if not ret:
        print("Training ended abnormally.")