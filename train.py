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
import torch
import numpy as np
from tqdm import tqdm
from agent import Agent
from agent_ppo import PPOAgent
from enviroment import FrameEnv
import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.get_state import cluster_train
from utils.yolov5.utils.general import print_args

# test 

def _save_q_table(qTable, filePath="models/q_table.npy") :
    print("save!")
    np.save(filePath, qTable)

def str2bool(v) :
    if isinstance(v, bool) :
        return v
    if v.lower() in ('true', 'yes', 't') :
        return True
    elif v.lower() in ('false', 'no', 'f') :
        return False
    else :
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_opt(known=False) :
    parser = argparse.ArgumentParser()
    parser.add_argument("-ei", "--epsilonInit", type=int, default=1, help="epsilon init value")
    parser.add_argument("-ed", "--epsilonDecreseRate", type=float, default=0.005, help="epsilon decrese value")
    parser.add_argument("-em", "--epsilonMinimum", type=float, default=0.1, help="epsilon minimum value")
    
    parser.add_argument("-f", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-w", "--window", type=int, default=30, help="importance calculate object detect range")
    parser.add_argument("-s", "--stateNum", type=int, default=15, help="clustering state Number")
    parser.add_argument("-lr", "--lr", type=int, default=0.05, help="setting learning rate")
    parser.add_argument("-priorC", "--isClusterexist", type=str2bool, default=True, help="using pretrained cluster model?")
    
    parser.add_argument("-vp", "--videoPath", type=str, default="data/RoadVideo-train.mp4", help="training video path") # using Jetson-video : "data/jetson-train.mp4"
    # parser.add_argument("-vn", "--videoName", type=str, default="jetson-train", help="setting video name")

    parser.add_argument("-priorD", "--isDetectionexist", type=str2bool, default=True, help= "using predetected txt file?")
    parser.add_argument("-drp", "--detectResultPath", type=str, default="utils/yolov5/runs/detect/exp/labels", help="detect file path")
    # parser.add_argument("-cp", "--clusterPath", type=str, default="models/cluster_jetson-train.pkl", help="cluster model path")
    
    # *** soft ***
    parser.add_argument("-sw", "--softWeight", type=float, default=0.9, help="weight for combine soft and Q")
    parser.add_argument("-t", "--threshold", type=float, default=0.05, help="init threshold for select feasible set")
    parser.add_argument("-td", "--threshold_decrese", type=float, default=0.0002, help="threshold decrease rate")
    parser.add_argument("-std", "--std", type=float, default=5, help="std for gaussian distribution")
    parser.add_argument("-pt", "--pType", type=int, default=1, help="1 is combined 2 is Gaussian")
    # *** require ***
    parser.add_argument("-qp", "--qTablePath", type=str, default="models/q_table", help="qtable path")
    parser.add_argument("-b", "--beta", type=float, default=1.35, help="sensitive for number of objects")  # using Jetson-video : 0.5
    parser.add_argument("-m", "--masking", type=str2bool, default=True, help="using masking?")
    parser.add_argument("-soft", "--isSoft", type=str2bool, default=False, help="using soft-constraint?")
    parser.add_argument("-con", "--isContinue", type=str2bool, default=False, help="for Jetson Training")
    parser.add_argument("-method", "--method", type=str, default="PPO", help="learning algorithm")
    parser.add_argument("-pipe", "--pipeNum", type=int, default=1, help="pipe")
    
    # *** PPO ***
    parser.add_argument("-mode", "--mode", type=str, default="train", help="train / val / test")
    parser.add_argument("-optlr", "--opt_learning_rate", type=float, default=0.0003, help="learning rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help=" discount factor gamma")
    parser.add_argument("-lmbda", "--lmbda", type=float, default=0.9, help="hyperparameter lambda for cal GAE")
    parser.add_argument("-clip", "--eps_clip", type=float, default=0.2, help="clip parameter for PPO")
    
    parser.add_argument("-episode", "--num_episode", type=int, default=500, help="number of train episode")
    parser.add_argument("-Kepoch", "--K_epochs", type=int, default=3, help="update policy for K Epoch")
    parser.add_argument("-buff", "--buffer_size", type=int, default=10, help="PPO buffer size")
    parser.add_argument("-rollout", "--rollout_len", type=int, default=320, help="i.e., training interval")
    parser.add_argument("-batch", "--minibatch_size", type=int, default=32, help="minibatch size")
    
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="hyperparameter alpha for cal Reward")

    parser.add_argument("-sdim", "--state_dim", type=int, default=2, help="state vector dimension")
    parser.add_argument("-anum", "--action_num", type=int, default=30, help="number of action (range)")
    return parser.parse_known_args()[0] if known else parser.parse_args()

def _main(opt, conf):
    # setting
    logdir="../total_logs/train/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)

    episode_maxlen = conf["num_episode"]
    videoName = opt.videoPath.split("/")[-1].split(".")[0]
    masking = opt.masking
    isContinue = opt.isContinue
    
    # detect result 편의를 위해
    if videoName == "jetson-train" :
        detectResultPath = "utils/yolov5/runs/detect/exp/labels"
    elif videoName == "RoadVideo-train" :
        detectResultPath = "utils/yolov5/runs/detect/exp2/labels"
    elif videoName == "jetson-train-new" :#TODO 
        detectResultPath = "utils/yolov5/runs/detect/exp3/labels"
    else :
        detectResultPath=opt.detectResultPath
        
    print_args(vars(opt))
    
    data_maxlen = 10000
    replayBuffer_len = 1000
    replayBuffer_maxlen = 20000
    qTablePath = opt.qTablePath
    isClusterexist = opt.isClusterexist
    
    clusterPath = "models/cluster_"+videoName+".pkl"
    envV = FrameEnv(videoName=videoName, videoPath=opt.videoPath, clusterPath=clusterPath, resultPath=detectResultPath, data_maxlen=data_maxlen, replayBuffer_maxlen=replayBuffer_maxlen, fps=opt.fps, w=opt.window, stateNum=opt.stateNum, isDetectionexist=opt.isDetectionexist, isClusterexist=isClusterexist, isRun=False, masking=masking, beta=opt.beta, runmode=opt.pipeNum, isSoft=opt.isSoft, method=opt.method)   # etc

    if opt.method == "Q-learning" :
        epi_actions = 500

        agentV = Agent(eps_init=opt.epsilonInit, eps_decrese=opt.epsilonDecreseRate, eps_min=opt.epsilonMinimum, fps=opt.fps, lr=opt.lr, gamma=opt.gamma, stateNum=opt.stateNum, threshold=opt.threshold, std=opt.std, pType=opt.pType, threshold_decrese=opt.threshold_decrese, isRun=False, masking=masking, isContinue=isContinue, isSoft=opt.isSoft)
        randAction = True
        for epi in range(episode_maxlen):       
            print("episode :", epi)
            done = False
            cluterVisualize = True
            showLog = False
            if epi == 0 or (epi % 50) == 0 or epi == episode_maxlen-1 :
                cluterVisualize = True
                showLog = True
            s = envV.reset(showLog=showLog)
            while not done:
                requireSkip = opt.fps - envV.targetA
                a = agentV.get_action(s, requireSkip, randAction)
                s, _, done = envV.step(a)
            if envV.buffer.size() > replayBuffer_len:
                randAction = False
                print("Q update ...")
                for _ in range(epi_actions):  # # of sampling count
                    trans = envV.buffer.get()
                    agentV.Q_update(trans)
                agentV.decrease_eps()
            # logging
            if isClusterexist :
                writer.add_scalar("Reward/one_Reward", envV.reward_sum, epi)
                writer.add_scalar("Network/Diff", (envV.ASum - envV.aSum), epi)
                writer.add_scalar("Network/target_A(t)", envV.ASum, epi)
                writer.add_scalar("Network/send_a(t)", envV.aSum, epi)
            if (epi % 50) == 0 or epi == episode_maxlen-1 :
                agentV.Q_show()
                qTable = agentV.get_q_table()
                if epi == episode_maxlen-1 :
                    _save_q_table(qTable, qTablePath+".npy")
                # else :
                #    _save_q_table(qTable, qTablePath+"("+str(epi)+").npy")
            if showLog :
                envV.trans_show()
            # after first episode, make cluster model
            if not isClusterexist and epi == 1:
                
                isClusterexist = True
            print("buffer size: ", envV.buffer.size())
            envV.omnet.get_omnet_message()
            envV.omnet.send_omnet_message("finish")
            print("sum of A(t) : ", envV.ASum, "| sum of a(t) : ", envV.aSum)
        envV.omnet.close_pipe()
        return agentV.get_q_table()
    
    
    if opt.method == "PPO" :
        rollout_len = conf["rollout_len"]
        agentV = PPOAgent(config=conf)
        total_reward = 0
        print_interval = 50
        for episode in tqdm(range(episode_maxlen)):
            epi_reward = 0
            state = envV.reset(showLog=True)
            done = False
            while not done :
                for t in range(rollout_len):
                    guide = opt.fps - envV.targetA
                    action, action_prob = agentV.get_actions(state, guide)
                    # print(action)
                    state_prime, reward, done = envV.step(action)
                    agentV.put_data((state, action, reward, state_prime, action_prob, done, guide)) 
                    epi_reward += reward
                    state = state_prime
                    if done : 
                        break
                loss, value_loss, policy_loss = agentV.train_net()
            total_reward += epi_reward
            
            # record total_reward & avg_reward & loss for each episode
            writer.add_scalar("Reward/one_Reward", epi_reward, episode)
            writer.add_scalar("Network/Diff", (envV.ASum - envV.aSum), epi)
            writer.add_scalar("Network/target_A(t)", envV.ASum, epi)
            writer.add_scalar("Network/send_a(t)", envV.aSum, epi)
            if loss is not None :
                writer.add_scalar("loss", loss.mean().item(), episode)
                writer.add_scalar("value_loss", sum(value_loss).mean().item(), episode)
                writer.add_scalar("policy_loss", sum(policy_loss).mean().item(), episode)

            if episode % print_interval == 0 :
                print("\n# of episode :{}, avg reward : {:.2f}, total reward : {:.2f}".format(episode, total_reward/print_interval, total_reward))
                total_reward = 0

            envV.omnet.get_omnet_message()
            envV.omnet.send_omnet_message("finish")
            print("sum of A(t) : ", envV.ASum, "| sum of a(t) : ", envV.aSum)
            
        envV.omnet.close_pipe()
        return 


if __name__ == "__main__":
    opt = parse_opt()
    conf = dict(**opt.__dict__)
    _main(opt, conf)
    print("Training Finish!")