import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class RolloutBuffer:
    def __init__(self, buffer_size, minibatch_size):
        self.buffer = []
        self.batch_data = []
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size


    def _make_batch(self):
        """ object : buffer에 저장된 buffer_size * minibatch_size개의 데이터를 buffer_size개의 minibatch가 담긴 데이터로 전환하여 저장합니다."""
        self.batch_data = []
        for i in range(self.buffer_size):
            s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch, guide_batch = [], [], [], [], [], [], []
            for j in range(self.minibatch_size):
                transition = self.buffer.pop()
                s, a, r, s_prime, prob_a, done, guide = transition
                s_batch.append(s)
                a_batch.append([a])
                r_batch.append([r])
                s_prime_batch.append(s_prime)
                prob_a_batch.append([prob_a])
                done_mask = 0 if done else 1
                done_batch.append([done_mask])
                guide_batch.append([guide])
            
            mini_batch = torch.tensor(s_batch),  torch.tensor(a_batch, dtype=torch.float), torch.tensor(r_batch, dtype=torch.float), \
                torch.tensor(s_prime_batch), torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float), torch.tensor(guide_batch, dtype=torch.float)

            self.batch_data.append(mini_batch)
    
    
    def clear(self):
        """ object: buffer를 비웁니다.
        input: None
        output: None
        """
        del self.buffer[:]
    
    
    def put(self, transition):
        """ object: buffer에 transition을 넣습니다.
        input: transition -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int, float], float, 
                Tuple[torch.Tensor, torch.Tensor], Tuple[float, float, float], bool]
        output: None
        """
        self.buffer.append(transition)
    
    
    def get_batch(self):
        """ object: 내부에서 _make_batch()를 호출하여 만든 batch data를 반환합니다.
        input: None
        output: self.batch_data -> List[[[List[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, 
                                        List[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor], ... (#32)], ... (#10)]
        """
        self._make_batch()
        
        return self.batch_data
        
        
    def is_full(self):
        """ object: 현재 buffer가 minibatch를 만들 수 있을 정도로 가득 찬 상태인지 확인하고 그 결과를 반환합니다.
        input: None
        output: bool
        """
        flag = False
        if len(self.buffer) >= self.buffer_size * self.minibatch_size :
            flag = True
        
        return flag


class PPOAgent(nn.Module):
    def __init__(self, config):
        super(PPOAgent, self).__init__()
        self.buffer = RolloutBuffer(config["buffer_size"], config["minibatch_size"])
        # parameter setting
        ### parameter for PPO
        self.mode = config["mode"]
        self.lr = config["opt_learning_rate"]
        self.gamma = config["gamma"]
        self.lmbda = config["lmbda"]
        self.eps_clip = config["eps_clip"]     
        self.K_epochs = config["K_epochs"]               
        self.minibatch_size = config["minibatch_size"]   # M
        ### parameter for Frame-drop RL
        self.masking = config["masking"]
        self.alpha = config["alpha"]
        self.state_dim = config["state_dim"]
        self.action_num = config["action_num"]
        
        # PPO model setting   
        self.shared_layer = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
        )
        if not self.masking :
            self.actor_mu = nn.Linear(128, 1)  # action dist mu
        self.actor_std = nn.Linear(128, 1)  # action dist std 
        self.critic = nn.Linear(128, 1) # v
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimization_step = 0


    def _policy(self, state, guide=-1, batch=False):
        """ object: state를 policy network에 통과시켜 얻은 mu, std로부터 action의 Normal distribution을 계산해 반환합니다."""
        x = self.shared_layer(state)

        if not self.masking :
            x_mu = self.actor_mu(x)
            if batch :
                x_mu = x_mu.transpose(0, 1)
            mu = self.action_num * torch.sigmoid(x_mu)
        else :
            mu = guide

        x_std = self.actor_std(x)
        if batch :
            x_std = x_std.transpose(0, 1)
        std = F.softplus(x_std)
        
        dist = Normal(mu, std)
        
        return dist
    
    
    def _calc_advantage(self, data):
        """ object: 하나의 batch 안에 들어있는 각각의 mini_batch별로 Advantage를 계산하고 advantage를 추가한 batch를 만들어 반환합니다."""
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_probs, guide = mini_batch
            with torch.no_grad():                
                td_target = r.view(-1, 1) + self.gamma * self.get_value(s_prime).view(-1, 1) * done_mask.view(-1, 1)
                delta = td_target - self.get_value(s).view(-1, 1)
                
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_probs, td_target, advantage, guide))

        return data_with_adv
    
    
    def get_value(self, state):
        """ object:  state를 value network에 통과시켜 얻은 value를 반환합니다.
        *get_value는 Advantage를 계산할 때, 항상 batch단위로 호출되기 때문에 output이 Tensor형태입니다.
        input: state -> Tuple[torch.Tensor, torch.Tensor]
        output: value -> torch.Tensor[float]
        """
        state = torch.tensor(state)
        state = state.float()
        x = self.shared_layer(state)
        v = self.critic(x)
        
        return v
    
    
    def get_actions(self, state, guide=-1, train=False):
        """ object: input을 받아, 내부에서 _policy 함수를 호출한 뒤 action과 해당 action의 log_prob들을 tuple로 반환합니다.
        *만약 train하는 과정이라면 batch 단위로 실행되기 때문에 output이 (32, .) 형태의 Tensor로 반환됩니다.
        input: state -> Tuple[float, float]; guide -> int; train -> bool
        output: actions -> Tuple[int, int, float]; probs -> Tuple[float, float, float]
        """
        state = torch.tensor(state)
        state = state.float()
        if not train:
            self.eval()
            with torch.no_grad():
                dist = self._policy(state, guide, batch=False)
        else :
            self.train()
            dist = self._policy(state, guide, batch=True)
        
        a = dist.sample()
        a_int = torch.clamp(a, 0, 30).int()
        log_prob = dist.log_prob(a)
        
        return a_int, log_prob


    def put_data(self, transition):
        """ object: 1번의 transition 데이터를 Tensor타입을 제거한 뒤 put()을 호출하여 buffer에 집어넣습니다.
        input: transition -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int, float], float, 
                                Tuple[torch.Tensor, torch.Tensor], Tuple[float, float, float], bool]
        output: None
        """
        s, a, r, s_prime, prob_a, done, guide = transition
        self.buffer.put((s, a, r, s_prime, prob_a, done, guide))

        
    def train_net(self):
        """ object: buffer가 가득차면 buffer에 쌓여있는 데이터를 사용하여 K_epochs번 DNN의 업데이트를 진행합니다.
        input: None
        output: None or loss
        """
        loss = None
        v_loss_list = []
        policy_loss_list = []
        if self.buffer.is_full() :
            data = self.buffer.get_batch()
            data = self._calc_advantage(data)

            for _ in range(self.K_epochs): 
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_probs, td_target, advantages, guide = mini_batch
                    old_log_probs = old_log_probs.transpose(0, 1)
                    actions, log_probs = self.get_actions(s, guide, train=True)

                    v_loss = F.smooth_l1_loss(self.get_value(s) , td_target)
                    v_loss_list.append(v_loss)
                    
                    ratio = torch.exp(log_probs - old_log_probs).view(-1, 1)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
                    policy_loss = -torch.min(surr1, surr2)
                    policy_loss_list.append(policy_loss)

                    # policy loss + value loss
                    loss = policy_loss + v_loss
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
        
        return loss, v_loss_list, policy_loss_list
                    