"""
RL Agent that drops unnecessary frames depending on the network state in the end device.
action : skip-length
"""
import numpy as np
import random
import copy
from typing import Dict, Union, List, Tuple

random.seed(42)
INF = float("inf")

class Agent():
    """Frame Dropping을 수행하는 강화학습 agent입니다.
    """
    def __init__(self, conf:Dict[str, Union[str, bool, int, float]], run:bool):
        """init: 학습에 필요한 초기설정

        Args:
            conf (Dict[str, Union[bool, int, float]]): train/testing setting
            run (bool): testing을 수행하는가?
        """
        self.state_num = conf['state_num']
        self.fps = conf['fps']
        self.masking = conf['is_masking']
        self.run = run
        self.omnet_mode = conf['omnet_mode']
        self.action_dim = conf['action_dim']
        
        if not self.run: 
            self.eps, self.eps_dec, self.eps_min = conf['eps_init'], conf['eps_dec'], conf['eps_min']
            self.lr, self.gamma = conf['learning_rate'], conf['gamma']
        
        if conf['is_continue'] or self.run : 
            self.qtable = self.__load_model(conf['model_path'])
        else :
            self.qtable = np.zeros((self.state_num, self.action_dim+1))  # s, a
        self.action_space = list(range(self.action_dim+1))  # 0 to fps
        self.isFirst = True


    def get_action(self, s:int, require_skip:int=-1, rand:bool=True) -> int : 
        """현재 state, 그리고 guide에 기반하여 skip이 필요한지를 전달받아 action을 선택합니다.

        Args:
            s (int): _description_
            require_skip (int, optional): _description_
            rand (bool, optional): _description_

        Returns:
            int: action
        """
        action = 0
        if self.isFirst and require_skip == self.action_dim :
            require_skip -= 1
            self.isFirst = False
        
        if self.run :
            temp = copy.deepcopy(self.qtable[s, :])
            temp_vec = temp.flatten()
            while True :
                action = np.argmax(temp_vec)
                if action >= require_skip :
                    break
                temp_vec[action] = (-1)*INF
        else :
            if rand :
                action = random.choice(self.action_space[:])
            else : 
                p = random.random()
                if p < self.eps :  
                    if self.masking : 
                        action = random.choice(self.action_space[require_skip:])
                    else : 
                        action = random.choice(self.action_space[:])
                else:  
                    temp = copy.deepcopy(self.qtable[s, :])
                    temp_vec = temp.flatten()
                    if self.masking : 
                        while True :
                            action = np.argmax(temp_vec)
                            if action >= require_skip :
                                break
                            temp_vec[action] = (-1)*INF
                    else :
                        action = np.argmax(temp_vec)
        return action


    def update_qtable(self, trans:Tuple[int, int, int, int, float]):
        """buffer에서 random sampling한 transition을 사용하여 qtable을 갱신합니다.

        Args:
            trans (Tuple[require_skip, state, action, next_state, reward])
        """
        if self.omnet_mode:
            require_skip, s, a, s_prime, r = trans
        else:
            s, a, s_prime, r = trans
        self.qtable[s, a] = self.qtable[s, a] + self.lr * \
            (r + self.gamma * np.max(self.qtable[s_prime, :]) - self.qtable[s, a])

        return


    
    def decrease_eps(self):
        """Q learning 학습을 위해 사전 설정한 값을 이용하여 eps-greedy에 사용하는 epsilon값을 변경합니다.
        """
        self.eps -= self.eps_dec
        self.eps = max(self.eps, self.eps_min)
    

    def __get_qtable(self) -> np.ndarray :
        """현재 agent의 qtable을 반환합니다.

        Returns:
            np.ndarray:self.qtable
        """
        return self.qtable


    def __load_model(self, save_path: str) -> bool:
        """저장된 경로에서 파일을 읽어와 qtable을 agent에 로드합니다.

        Args:
            save_path (str): 모델이 저장된 경로

        Returns:
            bool: 해당 경로에서 불러온 qtable
        """
        qtable = np.load(save_path)
        
        return qtable


    def save_model(self, save_path: str) -> bool:
        """현재 agent의 qtable을 지정된 경로에 저장합니다.

        Args:
            save_path (str):저장경로

        Returns:
            bool: 저장이 정상적으로 수행되었으면 True
        """
        qtable = self.__get_qtable()
        np.save(save_path, qtable)
    
        return True


    def show_qtable(self):
        """현재 agent의 qtable을 command line에 출력합니다.
        """
        for s in range(self.state_num) :
            for a in range(self.action_dim+1) :
                print(round(self.qtable[s][a], 2), end="   ")
            print()