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

ARRIVAL_MAX = 1.0

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


class VideoProcessor():
    """영상을 프레임단위로 읽어들이거나 내보내기 위한 관리자
    """
    def __init__(self, video_path: str, fps: int, action_dim: int, output_path: str=None, f1_score: bool=True, write:bool=False) :
        """init
        
        Args:
            video_path: 학습/테스트에 사용할 영상의 경로
            fps: 영상의 fps
            output_path: 테스트시 skip한 영상을 내보낼 경로
            f1_score: f1_score 테스트를 위한 영상인가?
            write: 처리된 영상을 생성할 것인가?
        """
        self.video_path = video_path
        self.output_path = output_path
        self.fps = fps
        self.action_dim = action_dim
        self.f1_score = f1_score
        self.write = write
        self.cap = cv2.VideoCapture(self.video_path)
        self.processed_frames = []
        self.num_all, self.num_processed = 0, 0
    
    
    def reset(self):
        """학습에 반복적으로 사용하기 위해 videocapture를 초기화한다.
        """
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.processed_frames = []
        _, f_init = self.cap.read()
        self.prev_frame, self.cur_frame = f_init, f_init
        self.idx = 0
        self.num_all, self.num_processed = 0, 0
    
    
    def read_video(self, skip:int) -> bool:
        """주어진 skip길이만큼 skip한 뒤, 나머지 영상은 저장한다.

        Args:
            skip (int): skip하는 프레임의 개수 [0, fps]

        Returns:
            bool: 영상이 끝났는가?
        """
        skip_frame = self.prev_frame
        for _ in range(skip):
            if self.f1_score and self.write:
                self.processed_frames.append(skip_frame)

            self.prev_frame = self.cur_frame
            ret, self.cur_frame = self.cap.read()
            self.idx += 1
            self.num_all += 1
            if not ret:
                return False
        
        for _ in range(self.action_dim - skip):
            if self.write:
                self.processed_frames.append(self.cur_frame)
            
            self.prev_frame = self.cur_frame
            ret, self.cur_frame = self.cap.read()
            self.idx += 1
            self.num_all += 1
            self.num_processed += 1
            if not ret:
                return False

        return True


    def write_video(self) -> bool:
        """저장한 processed_frame들을 사용하여 영상으로 내보낸다.

        Returns:
            bool: 영상 저장이 성공적으로 이루어졌는가?
        """
        frame_shape = self.processed_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (frame_shape[1], frame_shape[0]))
        for frame in self.processed_frames:
            out.write(frame)

        out.release()
        
        return os.path.exists(self.output_path)
    
    
    def get_frame(self) -> Tuple[np.array, np.array, int]:
        """현재 frame과 이전 frame, 그리고 현재 frame의 idx를 반환한다.

        Returns:
            Tuple[prev_frame, cur_frame, idx]:
        """
        return self.prev_frame, self.cur_frame, self.idx


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
        self.fps = conf['fps']
        self.learn_method = conf['learn_method']
        self.V = conf['V']
        self.debug_mode = conf['debug_mode']
        
        if conf['pipe_num'] == 1:
            self.omnet = Communicator("\\\\.\\pipe\\frame_drop_rl", 200000, self.debug_mode)
        else :
            self.omnet = Communicator("\\\\.\\pipe\\frame_drop_rl_"+str(conf['pipe_num']), 200000, self.debug_mode)
        
        if self.run:
            self.video_processor = VideoProcessor(conf['video_path'], conf['fps'], conf['output_path'], conf['f1_score'], write=True)
        else:
            self.video_processor = VideoProcessor(conf['video_path'], conf['fps'])
            self.buffer = ReplayBuffer(conf['buffer_size'])
            self.beta = conf['beta']
            self.__detect(conf['detection_path'], conf['FFT_path'])
        
        if self.learn_method == "Q" :
            self.model = cluster_init(conf['state_num'])
            print("load cluster model in init...")
            self.model = cluster_load(conf['cluster_path'])
        
        self.reset()

    
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
        self.video_processor.reset()
        self.reward_sum, self.sum_A, self.sum_a = 0, 0, 0
        self.show_log = show_log
        self.prev_A, self.target_A = self.fps, self.fps
        self.prev_frame, self.cur_frame, self.idx = self.video_processor.get_frame()
        
        if self.run :
            self.origin_state = [0, get_FFT(self.cur_frame)]
        else :
            self.origin_state = [0, self.FFT_list[self.idx]]
        
        if self.learn_method == "Q" :
            self.state = cluster_pred(self.origin_state, self.model)
        elif self.learn_method == "PPO" :
            self.state = self.origin_state
        
        self.trans_list = []
        if self.show_log :   
            self.logList = []
            
        return self.state


    def __detect(self, detection_path: str, FFT_path: str):
        """학습에 사용하는 영상에 해당되는 사전처리된 데이터 로드

        Args:
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
        #print("\naction:", action)
        done = False
        ret = self.video_processor.read_video(skip=action)
        if not ret:
            done = True
            r = 0
            if self.run:
                create = self.video_processor.write_video()
                if create:
                    print("Writing the video has successfully concluded.")
                else:
                    print("Writing the video ended abnormally.")
            
            return self.state, r, done
        
        for a in range(self.fps+1):
            if a == action :
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("action")
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message(str((action+1)/self.fps))
            elif a > action : 
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message("action")
                self.omnet.get_omnet_message()
                self.omnet.send_omnet_message(str((1)/self.fps))
        
        
        self.prev_frame, self.cur_frame, self.idx = self.video_processor.get_frame()

        self.omnet.get_omnet_message()
        self.omnet.send_omnet_message("reward") 
        path_cost = float(self.omnet.get_omnet_message())
        self.omnet.send_omnet_message("ACK")
        
        ratio_A = ARRIVAL_MAX if path_cost == 0 else min(ARRIVAL_MAX, self.V / path_cost)
        new_A = math.floor(ratio_A*(self.fps))
        if self.debug_mode:
            print("scaling cost using V:", ratio_A)
            print("arrival rate using fps:", new_A)
        
        r = self.__triggered_by_guide(new_A, action)
        
        return self.state, r, done


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
        r = 0
        self.send_A = self.fps - action
        self.sum_a += self.send_A
        self.sum_A += self.target_A
        self.trans_list.append(self.fps-self.target_A)
        self.trans_list.append(self.state)
        self.trans_list.append(action)
        self.prev_A = self.target_A
        self.target_A = new_A
        
        if self.run :
            self.origin_state = [get_MSE(self.prev_frame, self.cur_frame), get_FFT(self.cur_frame)]
        else :
            self.origin_state = [get_MSE(self.prev_frame, self.cur_frame), self.FFT_list[self.idx]]
    
        if self.learn_method == "Q" :
            self.state = cluster_pred(self.origin_state, self.model)
        elif self.learn_method == "PPO" :
            self.state = self.origin_state
        
        self.trans_list.append(self.state)
        
        if not self.run :
            r = self.__reward_function()
            self.buffer.put_data(self.trans_list)
        self.trans_list = []
        
        return r


    def __cal_important(self) -> List[float]:
        """현재 idx에서 1초동안의 프레임 (=fps)을 이용하여 물체 개수로부터 중요도를 계산합니다.
        
        Returns: List[important_score]
        """
        # get importance
        std_idx = self.idx - self.fps
        important_list = []
        imp_max = 1
        imp_min = -1 * self.beta
        max_num = max(self.obj_num_list[std_idx : self.idx+1])
        for f in range(self.fps) :
            importance = (self.obj_num_list[std_idx+f] / max_num) if max_num != 0 else 0
            #normalized_importance = (imp_max - imp_min)*(importance) + imp_min
            #important_list.append(normalized_importance)
            important_list.append(importance)

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
        self.reward_sum += r
        if self.show_log :
            self.logList.append("s(t): {:2d}\tu(t): {:2d}\ts(t+1): {:2d}\tr(t): {:.5f}\nA(t): {:2d}\tA(t+1): {:2d}".format(s[0], a, s_prime[0], r, self.prev_A, self.target_A))
            
        return r


    def show_trans(self) :
        """show_log가 true일 때, 에피소드 전체에서 각각의 transition을 모두 출력합니다.
        """
        for row in self.logList :
            print(row)
        return


class Environment_withoutNET():
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
        self.fps = conf['fps']
        self.learn_method = conf['learn_method']
        self.V = conf['V']
        self.debug_mode = conf['debug_mode']
        
        self.radius = conf['radius']
        self.state_num = conf['state_num']
        self.action_dim = conf['action_dim']
        
        if self.run:
            self.video_processor = VideoProcessor(conf['video_path'], conf['fps'], conf['action_dim'], conf['output_path'], conf['f1_score'], write=True)
        else:
            # training (reward) argument
            self.reward_method = conf['reward_method']
            self.important_method = conf['important_method']
            self.beta = conf['beta']
            self.window = conf['window']
            self.video_processor = VideoProcessor(conf['video_path'], conf['fps'], conf['action_dim'])
            self.buffer = ReplayBuffer(conf['buffer_size'])
            self.__detect(conf['detection_path'], conf['FFT_path'])
        
        if self.learn_method == "Q" :
            self.model = cluster_init(conf['state_num'])
            print("load cluster model in init...")
            self.model = cluster_load(conf['cluster_path'])
        
        self.reset()

    
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
        self.video_processor.reset()
        self.reward_sum, self.sum_a = 0, 0
        self.show_log = show_log
        self.prev_frame, self.cur_frame, self.idx = self.video_processor.get_frame()
        
        if self.run :
            self.origin_state = [0, get_FFT(self.cur_frame, self.radius)]
        else :
            self.origin_state = [0, self.FFT_list[self.idx]]
        
        if self.learn_method == "Q" :
            self.state = cluster_pred(self.origin_state, self.model)
        elif self.learn_method == "PPO" :
            self.state = self.origin_state
        
        self.trans_list = []
        if self.show_log :   
            self.logList = []
            
        return self.state


    def __detect(self, detection_path: str, FFT_path: str):
        """학습에 사용하는 영상에 해당되는 사전처리된 데이터 로드

        Args:
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
        #print("\naction:", action)
        done = False
        ret = self.video_processor.read_video(skip=action)
        if not ret:
            done = True
            r = 0
            if self.run:
                create = self.video_processor.write_video()
                if create:
                    print("Writing the video has successfully concluded.")
                else:
                    print("Writing the video ended abnormally.")
            
            return self.state, r, done
        
        self.prev_frame, self.cur_frame, self.idx = self.video_processor.get_frame()
        r = self.__triggered_by_guide(action)
        
        return self.state, r, done


    def __triggered_by_guide(self, action:int) -> float:
        """action에 대한 reward를 계산합니다.

        Args:
            action (int): 수행하고자 했던 action

        Returns:
            float: reward
        
        Calls:
            utils.cal_quality.get_MsE/get_FFT: state 계산을 위해 호출 (origin)
            utils.get_state.cluster_pred: state 계산을 위해 호출 (int)
            self.__reward_function: reward 계산
            
        """
        r = 0
        self.send_A = self.action_dim - action
        self.sum_a += self.send_A
        self.trans_list.append(self.state)
        self.trans_list.append(action)
        
        if self.run :
            self.origin_state = [get_MSE(self.prev_frame, self.cur_frame), get_FFT(self.cur_frame, self.radius)]
        else :
            self.origin_state = [get_MSE(self.prev_frame, self.cur_frame), self.FFT_list[self.idx]]
    
        if self.learn_method == "Q" :
            self.state = cluster_pred(self.origin_state, self.model)
        elif self.learn_method == "PPO" :
            self.state = self.origin_state
        
        self.trans_list.append(self.state)
        
        if not self.run :
            r = self.__reward_function()
            self.buffer.put_data(self.trans_list)
        self.trans_list = []
        
        return r

    def __cal_important(self) -> List[float]:
        """현재 idx에서 1초동안의 프레임 (=fps)을 이용하여 물체 개수로부터 중요도를 계산합니다.
        
        Returns: List[important_score]
        """
        # get importance
        std_idx = self.idx - self.action_dim
        important_list = []
        imp_max = 1
        imp_min = -1 * self.beta
        max_num = max(self.obj_num_list[std_idx : self.window+1])
        for f in range(self.action_dim) :
            importance = (self.obj_num_list[std_idx+f] / max_num) if max_num != 0 else 0
            #normalized_importance = (imp_max - imp_min)*(importance) + imp_min
            #important_list.append(normalized_importance)
            important_list.append(importance)
        
        return important_list


    def __reward_function(self):
        """reward를 계산합니다.

        Returns:
            float: reward
            
        Calls:
            self.__cal_important: 리워드 계산을 위해 각 프레임의 중요도를 계산합니다. 

        """
        important_list = self.__cal_important()
        s, a, s_prime = self.trans_list
        plusdiv = len(important_list[a+1:])
        minusdiv = len(important_list[:a+1]) 
        if plusdiv == 0 :
            plusdiv = 1
        if  minusdiv == 0:
            minusdiv = 1
        r = (sum(important_list[a+1:])/plusdiv) - (sum(important_list[:a+1])/minusdiv) 
        
        self.trans_list.append(r)
        self.reward_sum += r
        if self.show_log :
            self.logList.append("s(t): {:2d}\tu(t): {:2d}\ts(t+1): {:2d}\tr(t): {:.5f}".format(s[0], a, s_prime[0], r))
        
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
    def __init__(self, pipeName:str, buffer_size:int, debug_mode:bool):
        self.pipeName = pipeName
        self.buffer_size = buffer_size
        self.pipe = self.init_pipe(self.pipeName, self.buffer_size)
        self.debug_mode = debug_mode


    def send_omnet_message(self, msg:str):
        if self.debug_mode:
            print("sending msg:", msg)
        win32file.WriteFile(self.pipe, msg.encode('utf-8'))  # wait unil complete reward cal & a(t)

        
    def get_omnet_message(self) -> str:
        response_byte = win32file.ReadFile(self.pipe, self.buffer_size)
        response_str = response_byte[1].decode('utf-8')
        if self.debug_mode:
            print("receive msg:", response_str)
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