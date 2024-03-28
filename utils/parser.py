import argparse
import re
import os
from typing import Tuple, List, Dict, Union
from utils.util import str2bool


def parse_common_args() :
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-video", "--video_path", type=str, default=None, help="training video path") # using Jetson-video : "data/jetson-train.mp4"
    parser.add_argument("-fps", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-mask", "--is_masking", type=str2bool, default=True, help="using masking?")
    parser.add_argument("-con", "--is_continue", type=str2bool, default=False, help="continue learning?")
    parser.add_argument("-learn", "--learn_method", type=str, default="Q", help="learning algorithm")

    parser.add_argument("-reward", "--reward_method", type=str, default="00", help="using which reward function")
    parser.add_argument("-important", "--important_method", type=str, default="000", help="using which important score")
    parser.add_argument("-b", "--beta", type=float, default=0.5, help="sensitive for number of objects")
    parser.add_argument("-w", "--window", type=int, default=30, help="used to calculate important score")
    parser.add_argument("-r", "--radius", type=int, default=60, help="used to calculate blurring score")
    parser.add_argument("-s", "--state_num", type=int, default=15, help="clustering state Number")
    parser.add_argument("-a", "--action_dim", type=int, default=30, help="skipping action Number")
    
    #TODO: hyperparameter: action, radius, beta, threshold, f1score, V
    parser.add_argument("-t", "--threshold", type=float, default=0.0, help="target value for reward +- (in 1)")
    parser.add_argument("-f", "--target_f1", type=float, default=0.7, help="target f1 score for reward +- (in 2)")
    
    parser.add_argument("-pipe", "--pipe_num", type=int, default=1, help="number of pipe that use to connect with omnet")
    parser.add_argument("-V", "--V", type=float, default=100000000, help="trade off parameter between stability & accuracy")
    
    parser.add_argument("-debug", "--debug_mode", type=str2bool, default=False, help="using debug tool?")
    parser.add_argument("-omnet", "--omnet_mode", type=str2bool, default=False, help="using omnet guide in RL run?")
    
    #MORE Exploitation
    parser.add_argument("-ed", "--eps_dec", type=float, default=0.01, help="epsilon decrese value") #change 0.005 -> 0.01
    parser.add_argument("-em", "--eps_min", type=float, default=0.1, help="epsilon minimum value")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05, help="setting learning rate")
    parser.add_argument("-episode", "--episode_num", type=int, default=150, help="number of train episode")
    
    #state define
    #? 0: basic / 1: last frame diff & FFT / 2: last frame diff & i-1 frame diff
    parser.add_argument("-state", "--state_method", type=int, default=0, help="state define method")
    
    args, unknown = parser.parse_known_args()
    return args, parser


def parse_train_args() -> Tuple[Dict[str, Union[str, bool, int, float]], Dict[str, Union[str, bool, int, float]]]:
    args, parser = parse_common_args() 
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help=" discount factor gamma")
    
    if args.learn_method == 'Q':
        parser.add_argument("-ei", "--eps_init", type=float, default=1.0, help="epsilon init value")
        parser.add_argument("-sb", "--start_buffer_size", type=int, default=3000, help="start train buffer size")
        parser.add_argument("-samp", "--sampling_num", type=int, default=1000, help="Q-learning update num")  #change 500 -> 1000
        parser.add_argument("-buff", "--buffer_size", type=int, default=30000, help="Replay buffer size")    #change 20000 -> 30000
    
    elif args.learn_method == 'PPO':
        parser.add_argument("-l", "--lmbda", type=float, default=0.9, help="hyperparameter lambda for cal GAE")
        parser.add_argument("-clip", "--eps_clip", type=float, default=0.2, help="clip parameter for PPO")
        parser.add_argument("-epochr", "--K_epochs", type=int, default=3, help="update policy for K Epoch")
        parser.add_argument("-rollout", "--rollout_len", type=int, default=320, help="i.e., training interval")
        parser.add_argument("-batch", "--minibatch_size", type=int, default=32, help="minibatch size")
        parser.add_argument("-s", "--state_dim", type=int, default=2, help="state vector dimension")
        parser.add_argument("-a", "--action_dim", type=int, default=30, help="number of action (range)")
        parser.add_argument("-buff", "--buffer_size", type=int, default=10, help="PPO rollout buffer size")
        parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="setting learning rate")
    else:
        raise ValueError("learn_method is must be Q or PPO.")
    
    args, unknown_args = parser.parse_known_args()
    default_args_dict = vars(parser.parse_args([]))
    custom_args_dict = vars(args)

    return custom_args_dict, default_args_dict


def parse_test_args() : 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-video", "--video_path", type=str, default=None, help="testing video path")
    parser.add_argument("-fps", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-model", "--model_path", type=str, default=None, help="trained model path")
    parser.add_argument("-mask", "--is_masking", type=str2bool, default=True, help="using lyapunov based guide?")
    parser.add_argument("-out", "--output_path", type=str, default=None, help="output video Path")
    parser.add_argument("-f1", "--f1_score", type=str2bool, default=True, help="showing f1 score")
    parser.add_argument("-print", "--print_network", type=str2bool, default=False, help="cmd print log")
    parser.add_argument("-r", "--radius", type=int, default=60, help="used to calculate blurring score")
    parser.add_argument("-s", "--state_num", type=int, default=15, help="clustering state Number")
    parser.add_argument("-a", "--action_dim", type=int, default=30, help="skipping action Number")
    
    # may... -i == __1 : 0에서 1사이 / 0.3? | -i == __0 : 0.0
    parser.add_argument("-t", "--threshold", type=float, default=0.0, help="target value for reward +- (in 1)")
    parser.add_argument("-f", "--target_f1", type=float, default=0.7, help="target f1 score for reward +- (in 2)")
    
    parser.add_argument("-pipe", "--pipe_num", type=int, default=1, help="number of pipe that use to connect with omnet")
    # model_1: 100000000 | SLN: 50 | YOLO: 
    parser.add_argument("-V", "--V", type=float, default=100000, help="trade off parameter between stability & accuracy")
    
    parser.add_argument("-debug", "--debug_mode", type=str2bool, default=False, help="debug tool")
    parser.add_argument("-omnet", "--omnet_mode", type=str2bool, default=False, help="using omnet guide in RL run?")

    # parser.add_argument("-log", "--log_name", type=str, default=None, help="log name")
    parser.add_argument("-net", "--network_name", type=str, default=None, help="network name")
    
    parser.add_argument("-rl", "--using_RL", type=str2bool, default=True, help="using RL model? if it is False, then every frame is sended.")
    
    #state define
    #? 0: basic / 1: last frame diff & FFT / 2: last frame diff & i-1 frame diff
    parser.add_argument("-state", "--state_method", type=int, default=0, help="state define method")
    
    
    return parser.parse_args()


def add_args(conf):
    model_path = conf['model_path']
    assert model_path is not None, "model_path is None."

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
    root_output = "./results/output/"
    
    video_path = conf['video_path']
    video_name = re.split(r"[/\\]", video_path)[-1].split(".")[0]
    
    model_path = conf['model_path']
    name = re.split(r"[/\\]", model_path)[-1][:-4]
    parts = name.split('_')
    cluster_video_name = ""
    
    log_name = str(start_time) + "_" +  re.split(r"[/\\]", conf['video_path'])[-1][:-4] + "_" + parts[0]
    name_list = ['rewardmethod', 'threshold', 'radius', 'epsdec', 'statemethod', 'actiondim']
    match_dict = {
        'rewardmethod': 'reward',
        'threshold': 'thresh',
        'statemethod' : 'state',
        'radius': 'r',
        'epsdec': 'e'
    }
    
    for i in range(1, len(parts), 2):
        key = parts[i]
        value = parts[i+1]
        if key == 'statenum': 
            conf['state_num'] = int(value)
        if key == 'videopath':
            cluster_video_name = value
        if key == 'actiondim':
            conf['action_dim'] = int(value)
        if key == 'radius':
            conf['radius'] = int(value) 
        if key in name_list:
            log_name += ("_" + match_dict[key] + "_" + str(value))

    if conf['state_method'] == 0:
        conf['cluster_path'] = os.path.join(root_cluster, cluster_video_name + "_" + str(conf['state_num']) + "_" + str(conf['radius']) + "_" + str(conf['state_method']) + ".pkl")
    elif conf['state_method'] == 1:
        conf['cluster_path'] = os.path.join(root_cluster, cluster_video_name + "_" + str(conf['state_num']) + "_" + str(conf['radius']) + "_" + str(conf['action_dim']) + "_" +  str(conf['state_method']) + ".pkl")
    elif conf['state_method'] == 2:
        conf['cluster_path'] = os.path.join(root_cluster, cluster_video_name + "_" + str(conf['state_num']) + "_" + str(conf['action_dim']) + "_" + str(conf['state_method']) + ".pkl")
    elif conf['state_method'] == 0:
        conf['cluster_path'] = os.path.join(root_cluster, cluster_video_name + "_" + str(conf['state_num']) + "_" + str(conf['state_method']) + ".pkl")

    log_name += "_mask_" + str(conf["is_masking"])
    
    if conf["network_name"] is not None:
        log_name += "_net_"+conf["network_name"]
    if conf["omnet_mode"] :
        log_name += "_agent_"+str(conf["pipe_num"])

    conf["log_path"] = os.path.join(root_log, log_name)
    conf["output_path"] = os.path.join(root_output, start_time + ".mp4")
    #log_path = os.path.join(root_log, start_time + "_" + output_name)
    
    return conf, conf["log_path"]