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
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.get_state import cluster_train, cluster_init
from utils.util import str2bool
from utils.cal_quality import get_FFT, get_MSE

from utils.yolov5.detect import inference


def parse_common_args() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--video_path", type=str, default=None, help="training video path") # using Jetson-video : "data/jetson-train.mp4"
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-b", "--beta", type=float, default=1.35, help="sensitive for number of objects")  # using Jetson-video : 0.5
    parser.add_argument("-mask", "--is_masking", type=str2bool, default=True, help="using masking?")
    parser.add_argument("-con", "--is_continue", type=str2bool, default=False, help="continue learning?")
    parser.add_argument("-learn", "--learn_method", type=str, default="Q", help="learning algorithm")
    parser.add_argument("-reward", "--reward_method", type=str, default=None, help="using reward function")
    parser.add_argument("-pipe", "--pipe_num", type=int, default=1, help="number of pipe that use to connect with omnet")

    args, unknown = parser.parse_known_args()
    return args, parser


def parse_args() -> Tuple[Dict[str, Union[str, bool, int, float]], Dict[str, Union[str, bool, int, float]]]:
    args, parser = parse_common_args() 
    
    parser.add_argument("-episode", "--episode_num", type=int, default=500, help="number of train episode")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help=" discount factor gamma")
    parser.add_argument("-w", "--window", type=int, default=30, help="importance calculate object detect range")
    
    #TODO 
    if args.learn_method == 'Q':
        parser.add_argument("-ei", "--eps_init", type=int, default=1, help="epsilon init value")
        parser.add_argument("-ed", "--eps_dec", type=float, default=0.005, help="epsilon decrese value")
        parser.add_argument("-em", "--eps_min", type=float, default=0.1, help="epsilon minimum value")
        parser.add_argument("-s", "--state_num", type=int, default=15, help="clustering state Number")
        parser.add_argument("-sb", "--start_buffer_size", type=int, default=1000, help="start train buffer size")
        parser.add_argument("-sampling", "--sampling_num", type=int, default=500, help="Q-learning update num")
        parser.add_argument("-buff", "--buffer_size", type=int, default=20000, help="Replay buffer size")
        parser.add_argument("-lr", "--learning_rate", type=int, default=0.05, help="setting learning rate")
    
    elif args.learn_method == 'PPO':
        parser.add_argument("-l", "--lmbda", type=float, default=0.9, help="hyperparameter lambda for cal GAE")
        parser.add_argument("-clip", "--eps_clip", type=float, default=0.2, help="clip parameter for PPO")
        parser.add_argument("-epochr", "--K_epochs", type=int, default=3, help="update policy for K Epoch")
        parser.add_argument("-rollout", "--rollout_len", type=int, default=320, help="i.e., training interval")
        parser.add_argument("-batch", "--minibatch_size", type=int, default=32, help="minibatch size")
        parser.add_argument("-s", "--state_dim", type=int, default=2, help="state vector dimension")
        parser.add_argument("-a", "--action_dim", type=int, default=30, help="number of action (range)")
        parser.add_argument("-buff", "--buffer_size", type=int, default=10, help="PPO rollout buffer size")
        parser.add_argument("-lr", "--learning_rate", type=int, default=0.0003, help="setting learning rate")
    else:
        raise ValueError("learn_method is must be Q or PPO.")
    
    args, unknown_args = parser.parse_known_args()
    default_args_dict = vars(parser.parse_args([]))
    custom_args_dict = vars(args)

    return custom_args_dict, default_args_dict


def path_manager(video_path: str, state_num: int) -> Tuple[str, str, str] :
    """ video path를 전달받아서 cluster, detection, FFT 경로를 지정하여 반환합니다.

    Args:
        video_path (str): 학습에 사용할 영상의 경로
        state_num (int): 학습에 사용하는 cluster의 개수
        
    Returns:
        Tuple[cluster_path, detection_path, FFT_path]
    """
    root_cluster, root_detection, root_FFT = "./models/cluster/", "./data/detect/", "./data/FFT/"
    video_name = re.split(r"[/\\]", video_path)[-1].split(".")[0]
    cluster_path = os.path.join(root_cluster, video_name + "_" + str(state_num) + ".pkl")
    detection_path = os.path.join(root_detection, video_name + "/labels")
    FFT_path = os.path.join(root_FFT, video_name + ".npy")
    
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
    skip_list = ['learn_method', 'pipe_num', 'fps', 'episode_num']
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
    if conf['learn_method'] == "Q" :
        root_save = os.path.join(root_save, "ndarray")
        save_path = os.path.join(root_save, name + ".npy")
    elif conf['learn_method'] == "PPO" :
        root_save = os.path.join(root_save, "weight")
        save_path = os.path.join(root_save, name + ".pt")
    
    return log_path, save_path


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
        root_detection =  "./data/detect/"
        video_name = re.split(r"[/\\]", conf['video_path'])[-1].split(".")[0]
        command = ["--weights", "models/yolov5s6.pt", "--source", conf['video_path'], "--project", root_detection, "--name", video_name, "--save-txt", "--save-conf", "--nosave"]
        inference(command)
        print("finish making detection files !\n")
    
    if not os.path.exists(FFT_path):
        print("start making blur level file ...")
        cap = cv2.VideoCapture(conf['video_path'])
        FFTList = []
        idx = 0
        while True:
            print("video 1/1", idx)
            ret, frame = cap.read()
            if not ret:
                break
            blur = get_FFT(frame)
            FFTList.append(blur)
            idx += 1
        cap.release()
        np.save(FFT_path, FFTList)
        print("finish making blur level file !\n")
    
    if not os.path.exists(cluster_path):
        print("start making cluster model ...")
        cluster = cluster_init(state_num=conf['state_num'])
        FFTList = np.load(FFT_path)
        data = []
        cap = cv2.VideoCapture(conf['video_path'])
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
        cluster = cluster_train(cluster, np.array(data), cluster_path)
        print("finish making cluster model !\n")


def main(conf: Dict[str, Union[str, bool, int, float]], default_conf: Dict[str, Union[str, bool, int, float]]) -> bool :
    """argument를 전달받아, 그 설정대로 강화학습을 수행합니다.

    Args:
        conf (Dict[str, Union[bool, int, float]]): train argument
        default_conf (Dict[str, Union[bool, int, float]]): train argument's default value

    Returns:
        bool: 정상적으로 학습이 종료되면 True를 반환합니다.
    """
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S") 
    cluster_path, detection_path, FFT_path = path_manager(conf['video_path'], conf['state_num'])
    verifier(conf, cluster_path, detection_path, FFT_path)
    log_path, save_path = logging_mannager(start_time, conf, default_conf)
    writer = SummaryWriter(log_path)
    conf['cluster_path'] = cluster_path
    conf['detection_path'] = detection_path
    conf['FFT_path'] = FFT_path
    prnt(conf)
    
    env = Environment(conf)
    
    if conf['learn_method'] == "Q" :
        agent = Agent(conf, run=False)
        rand = True
        print("train start !")
        for epi in tqdm(range(conf['episode_num'])):       
            done = False
            show_log = False
            if epi == 0 or (epi % 50) == 0 or epi == conf['episode_num']-1 :
                show_log = True
            s = env.reset(show_log=show_log)
            while not done:
                require_skip = conf['fps'] - env.target_A
                a = agent.get_action(s, require_skip, rand)
                s, _, done = env.step(a)
            if env.buffer.get_size() < conf['start_buffer_size']:
                print("buffer size: ", env.buffer.get_size())
            else :
                rand = False
                for _ in range(conf["sampling_num"]):  # # of sampling count
                    trans = env.buffer.get_data()
                    agent.update_qtable(trans)
                agent.decrease_eps()
            # logging
            writer.add_scalar("Reward/"+conf['reward_method'], env.reward_sum, epi)
            writer.add_scalar("Network/Diff", (env.sum_A - env.sum_a), epi)
            writer.add_scalar("Network/target_A(t)", env.sum_A, epi)
            writer.add_scalar("Network/send_a(t)", env.sum_a, epi)
            print("sum of A(t) : ", env.sum_A, "| sum of a(t) : ", env.sum_a)
            if epi == 0 or (epi % 50) == 0 or epi == conf["episode_num"]-1 :
                agent.show_qtable()
                env.show_trans()
            
            env.omnet.get_omnet_message()
            env.omnet.send_omnet_message("finish")
    
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
    env.omnet.close_pipe()
    ret = agent.save_model(save_path)
    if not ret:
        print("Saving model ended abnormally")
    
    print("Training Finish with...")
    prnt(conf)
    print("\n✱ start time :\t", start_time)
    print("✱ finish time:\t", finish_time)
    
    return True


if __name__ == "__main__":
    conf, default_conf = parse_args()
    ret = main(conf, default_conf)
    
    if not ret:
        print("Training ended abnormally.")