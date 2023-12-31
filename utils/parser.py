import argparse
import re
import os
from typing import Tuple, List, Dict, Union
from utils.util import str2bool


def parse_common_args() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--video_path", type=str, default=None, help="training video path") # using Jetson-video : "data/jetson-train.mp4"
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-b", "--beta", type=float, default=1.35, help="sensitive for number of objects")  # using Jetson-video : 0.5
    parser.add_argument("-mask", "--is_masking", type=str2bool, default=True, help="using masking?")
    parser.add_argument("-con", "--is_continue", type=str2bool, default=False, help="continue learning?")
    parser.add_argument("-learn", "--learn_method", type=str, default="Q", help="learning algorithm")
    parser.add_argument("-reward", "--reward_method", type=str, default="default", help="using reward function")
    parser.add_argument("-pipe", "--pipe_num", type=int, default=1, help="number of pipe that use to connect with omnet")

    args, unknown = parser.parse_known_args()
    return args, parser


def parse_train_args() -> Tuple[Dict[str, Union[str, bool, int, float]], Dict[str, Union[str, bool, int, float]]]:
    args, parser = parse_common_args() 
    
    parser.add_argument("-episode", "--episode_num", type=int, default=500, help="number of train episode")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help=" discount factor gamma")
    
    #TODO 
    if args.learn_method == 'Q':
        parser.add_argument("-ei", "--eps_init", type=float, default=1.0, help="epsilon init value")
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


def parse_test_args() : 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--video_path", type=str, default=None, help="testing video path")
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-model", "--model_path", type=str, default=None, help="trained model path")
    parser.add_argument("-mask", "--is_masking", type=str2bool, default=True, help="using lyapunov based guide?")
    parser.add_argument("-out", "--output_path", type=str, default=None, help="output video Path")
    parser.add_argument("-f1", "--f1_score", type=str2bool, default=True, help="showing f1 score")
    parser.add_argument("-log", "--log_network", type=str2bool, default=False, help="cmd print log")
    
    return parser.parse_args()


def add_args(conf):
    model_path = conf['model_path']
    method = re.split(r"[/\\]", model_path)[-2].split(".")[0]
    if method == 'weight':
        conf['learn_method'] = 'PPO'
    else:
        conf['learn_method'] = 'Q'
    
    if conf['learn_method'] == 'Q':
        default_values = {
            'is_continue': False,
            'pipe_num': 1,
            'state_num': 15,
        }
        for key in default_values:
            if conf.get(key) is None:
                conf[key] = default_values[key]
    return conf


def parse_test_name(conf:Dict[str, Union[str, int, bool, float]], start_time:str) -> Tuple[Dict[str, Union[str, int, bool, float]], str]:
    """모델 경로에 기록된 각종 정보를 통해 conf를 설정합니다.

    Args:
        conf (Dict[str, Union[str, int, bool, float]]): parse_args로 전달받은 기본 설정

    Returns:
        Tuple[Dict[str, Union[str, int, bool, float]], str]: model_path parsing으로 분석한 설정이 추가된 dict, log_path
    """
    root_log = "./results/logs/test/"
    root_cluster = "./models/cluster/"
    
    model_path = conf['model_path']
    name = re.split(r"[/\\]", model_path)[-1].split(".")[0]
    parts = name.split('_')
    cluster_video_name = ""
    
    for i in range(1, len(parts), 2):
        key = parts[i]
        value = parts[i+1]
        if key == 'statenum':  #TODO PPO 
            conf['state_num'] = int(value)
        if key == 'videopath':
            cluster_video_name = value
    
    conf['cluster_path'] = os.path.join(root_cluster, cluster_video_name + "_" + str(conf['state_num']) + ".pkl")
    
    log_path = os.path.join(root_log, start_time + "_" + conf['output_path'])
    
    return conf, log_path