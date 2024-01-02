"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import collections
import random
import cv2
import os
import math
import numpy as np
import win32pipe, win32file
from typing import Tuple, List, Union, Dict
from utils.get_state import cluster_pred, cluster_load, cluster_init
from utils.cal_quality import get_FFT, get_MSE
from utils.cal_F1 import get_F1

random.seed(42)


class ReplayBuffer():
    """Q learning 학습을 위해 transition을 저장하는 버퍼
    """
    def __init__(self, buffer_size:int):
        """init
        
        Args:
            buffer_size (int): 버퍼에 저장가능한 총 transitino 개수
        """
        self.buffer = collections.deque(maxlen=buffer_size)
    
    
    def put_data(self, trans:Tuple[Union[int, Tuple[float, float]], int, float, Union[int, Tuple[float, float]], float, bool, int]):
        """인자로 전달된 transition을 버퍼에 넣는다.
        
        Args:
            trans (Tuple[state, action, reward, state_prime, action_prob, done, guide])
        """
        self.buffer.append(trans)
    
    
    def get_data(self):
        """버퍼에 저장된 데이터를 랜덤하게 하나 꺼낸다.
        
        Returns:
            trans (Tuple[state, action, reward, state_prime, action_prob, done, guide])
        """
        trans = random.choice(self.buffer)  # batch size = 1
        return trans
    
    
    def get_size(self) -> int:
        """햔재 버퍼에 쌓인 데이터의 개수를 반환한다.
        
        Returns:
            int: length of buffer
        """
        return len(self.buffer)


class Environment():
    """강화학습 agent와 상호작용하는 environment
    """
    def __init__(self, conf:Dict[str, Union[str, bool, int, float]], run:bool=False):
        """init

        Args:
            conf (Dict[str, Union[bool, int, float]]): train/test setting
            run (bool, optional): testing을 수행하는가?
        
        Calls:
            self.__detect(): obj, fft load
            utils.get_state.cluster_init(): cluster 초기화
            utils.get_state.cluster_load(): cluster_path에서 불러옴
            self.reset(): Env reset
        """
        self.run = run
        self.learn_method = conf['learn_method']
        # instance define
        self.buffer = ReplayBuffer(conf['buffer_size'])
        if conf['pipe_num'] == 1:
            self.omnet = Communicator("\\\\.\\pipe\\frame_drop_rl", 200000)
        else :
            self.omnet = Communicator("\\\\.\\pipe\\frame_drop_rl_"+str(conf['pipe_num']), 200000)
        # hyper parameter setting
        self.beta, self.w = conf['beta'], conf['window']
        # data load
        self.video_path = conf['video_path']
        self.fps = conf['fps']
        if not self.run :
            self.__detect(conf['cluster_path'], conf['detection_path'], conf['FFT_path'])
        if self.learn_method == "Q" :
            self.model = cluster_init(conf['state_num'])
            print("load cluster model in init...")
            self.model = cluster_load(conf['cluster_path'])
        self.sum_A = 0
        self.reset()
        # run
        if self.run :
            self.all_processed_frame_list = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_path = conf['output_path']
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame.shape[1], self.frame.shape[0]))
    
    
    def reset(self, show_log:bool=False) -> Union[int, List[float]]:
        """Env reset: Arrival data, reward, idx, frame_reader, state 초기화 

        Args:
            show_log (bool, optional): command line에 이번 에피소드 전체의 transition을 출력할 것인가?

        Returns:
            Union[int, List[float, float]]: initial state를 반환한다.
            
        Calls:
            utils.cal_quality.get_FFT(): run 모드일 때 FFT 계산을 위해 사용한다.
            utils.get_state.cluster_pred: train, Q-learning에서 state를 구하기 위해 사용한다.
        """
        # idx init
        if self.sum_A != 0 :
            self.cap.release()
        self.reward_sum, self.skip_time, self.sum_A, self.sum_a, self.cur_idx, self.cap_idx = 0, 0, 0, 0, 0, 0
        self.show_log = show_log
        self.cap = cv2.VideoCapture(self.video_path)
        self.prev_A, self.target_A = self.fps, self.fps
        # frame/state init
        _, f1 = self.cap.read()
        self.frame_list = [f1]
        self.processed_frame_list = []
        self.prev_frame = f1
        self.frame = f1
        if self.run :
            self.origin_state = [0, get_FFT(self.frame)]
        else :
            self.origin_state = [0, self.FFT_list[self.cur_idx]]
        
        if self.learn_method == "Q" :
            self.state = cluster_pred(self.origin_state, self.model)
        elif self.learn_method == "PPO" :
            self.state = self.origin_state
        # transition init
        self.trans_list = []
        if self.show_log :   
            self.logList = []
            
        return self.state


    def __detect(self, cluster_path: str, detection_path: str, FFT_path: str):
        """학습에 사용하는 영상에 해당되는 사전처리된 데이터 로드

        Args:
            cluster_path (str): 검증된 cluster_path
            detection_path (str): 검증된 detection_path 경로
            FFT_path (str): 검증된 cluster_path
        """
        self.FFT_list = np.load(FFT_path)
        label_list =  os.listdir(detection_path)
        self.obj_num_list = []

        for label in label_list :  # TODO 없던 물체 분간...?
            label_path = detection_path+"/"+label
            with open(label_path, 'r') as file :
                lines = file.readlines()
                self.obj_num_list.append(len(lines))


    def step(self, action:int) -> Tuple[Union[int, List[float]], float, bool]:
        """인자로 전달된 action을 수행하고, 행동에 대한 reward와 함께 next_state를 반환합니다..

        Args:
            action (int): 수행할 action [0, fps]

        Returns:
            Tuple[next_state, reward, done]
        
        Calls:
            self.Communicator.get/sent_omnet_message: omnet 통신
            __triggered_by_guideL Lyapunov based guide가 새로 들어왔을 때 변수 갱신, 리워드 계산을 위해 호출
        """
        # skipping
        if action == 0 : 
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message("action")
            self.omnet.get_omnet_message()
            self.omnet.send_omnet_message(str((self.skip_time+1)/self.fps))
            self.processed_frame_list.append(self.frame)
            self.skip_time = 0
        if action == self.fps :
            self.skip_time += self.fps
        
        for a in range(1, self.fps):
            self.cap_idx += 1
            ret, temp = self.cap.read()
            if not ret:
                self.cap.release()
                if self.run :
                    self.all_processed_frame_list += self.processed_frame_list
                    for frame in self.all_processed_frame_list :
                        self.out.write(frame)
                    self.out.release()
                return -1, 0, True
            
            self.frame_list.append(temp)
            if a == action :
                self.processed_frame_list.append(temp)
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("action")
                self.omnet.get_omnet_message()
                send_action = (self.skip_time+action+1)/self.fps
                self.omnet.send_omnet_message(str(send_action))
                self.skip_time = 0
            elif a > action : 
                self.processed_frame_list.append(temp)
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("action")
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message(str((1)/self.fps))
        
        self.prev_frame = self.frame_list[-1]
        ret, self.frame = self.cap.read()
        self.cap_idx += 1
        if not ret:
            self.cap.release()
            if self.run :
                for frame in self.all_processed_frame_list :
                    self.out.write(frame)
                self.out.release()
            
            return -1, 0, True
        
        self.omnet.get_omnet_message()
        self.omnet.send_omnet_message("reward") 
        ratio_A = float(self.omnet.get_omnet_message())
        self.omnet.send_omnet_message("ACK")
        
        new_A = math.floor(ratio_A*(self.fps))
        r = self.__triggered_by_guide(new_A, action)
        
        return self.state, r, False


    def __triggered_by_guide(self, new_A:int, action:int) -> float:
        """Lyapunov based guide가 전달되었을 때, 갱신하고 reward를 계산합니다.

        Args:
            new_A (int): 새로운 lyapunov based guide
            action (int): 수행하고자 했던 action

        Returns:
            float: reward
        
        Calls:
            utils.cal_quality.get_MsE/get_FFT: state 계산을 위해 호출 (origin)
            utils.get_state.cluster_pred: state 계산을 위해 호출 (int)
            self.__reward_function: reward 계산
            
        """
        self.send_A = self.fps - action
        self.sum_a += self.send_A
        self.sum_A += self.target_A
        self.trans_list.append(self.fps-self.target_A)
        self.trans_list.append(self.state)
        self.trans_list.append(action)
        self.frame_list = [self.frame]
        self.processed_frame_list = [self.frame]
        self.prev_A = self.target_A
        self.target_A = new_A
        
        self.data.append(self.origin_state)
        # new state
        if self.run :
            self.all_processed_frame_list += self.processed_frame_list
            self.origin_state = [get_MSE(self.prev_frame, self.frame), get_FFT(self.frame)]
        else :
            self.origin_state = [get_MSE(self.prev_frame, self.frame), self.FFT_list[self.cur_idx+self.fps]]
    
        if self.learn_method == "Q" :
            self.state = cluster_pred(self.origin_state, self.model)
        elif self.learn_method == "PPO" :
            self.state = self.origin_state
        
        self.trans_list.append(self.state)
        # reward
        if not self.run :
            r = self.__reward_function()
        self.buffer.put(self.trans_list)
        self.trans_list = []
        
        return r


    def __cal_important(self) -> List[float]:
        """현재 idx에서 1초동안의 프레임 (=fps)을 이용하여 물체 개수로부터 중요도를 계산합니다.
        
        Returns: List[important_score]
        """
        # get importance
        important_list = []
        imp_max = 1
        imp_min = -1 * self.beta
        max_num = max(self.obj_num_list[self.cur_idx : self.cur_idx+self.fps+1])
        for f in range(self.fps) :
            importance = (self.obj_num_list[self.cur_idx+f] / max_num) if max_num != 0 else 0
            normalized_importance = (imp_max - imp_min)*(importance) + imp_min
            important_list.append(normalized_importance)
        
        return important_list


    def __reward_function(self):
        """reward를 계산합니다.

        Returns:
            float: reward
            
        Calls:
            self.__cal_important: 리워드 계산을 위해 각 프레임의 중요도를 계산합니다. 

        """
        important_list = self.__cal_important()
        _, s, a, s_prime = self.trans_list
        plusdiv = len(important_list[a+1:])
        minusdiv = len(important_list[:a+1]) 
        if plusdiv == 0 :
            plusdiv = 1
        if  minusdiv == 0:
            minusdiv = 1
        r = (sum(important_list[a+1:])/plusdiv) - (sum(important_list[:a+1])/minusdiv) 
        
        self.trans_list.append(r)
        self.cur_idx += self.fps
        self.reward_sum += r
        if self.show_log :
            self.logList.append("A(t): "+str(self.prev_A)+" action: "+str(a)+" A(t+1): "+str(self.target_A)+" reward: "+str(r))
        return r


    def show_trans(self) :
        """show_log가 true일 때, 에피소드 전체에서 각각의 transition을 모두 출력합니다.
        """
        for row in self.logList :
            print(row)
        return

class Communicator(Exception):
    """imnet과의 통신을 위해 사용합니다.

    Args:
        Exception (_type_)
    """
    def __init__(self, pipeName:str, buffer_size:int):
        self.pipeName = pipeName
        self.buffer_size = buffer_size
        self.pipe = self.init_pipe(self.pipeName, self.buffer_size)


    def send_omnet_message(self, msg:str):
        win32file.WriteFile(self.pipe, msg.encode('utf-8'))  # wait unil complete reward cal & a(t)

        
    def get_omnet_message(self) -> str:
        response_byte = win32file.ReadFile(self.pipe, self.buffer_size)
        response_str = response_byte[1].decode('utf-8')
        return response_str


    def close_pipe(self):
        win32file.CloseHandle(self.pipe)


    def init_pipe(self, pipeName:str, buffer_size:int) -> int :
        pipe = None
        print(pipeName)
        print("waiting connect OMNeT++ ...")
        try:
            pipe = win32pipe.CreateNamedPipe(
                pipeName,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1,
                buffer_size,
                buffer_size,
                0,
                None
            )
        except:
            print("except : pipe error")
            return -1

        win32pipe.ConnectNamedPipe(pipe, None)
        print("finshing connect OMNeT++")
        return pipe