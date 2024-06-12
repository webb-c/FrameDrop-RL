from collections import Counter
import csv
import os
from utils.yolov5.detect import inference
from utils.cal_F1 import get_F1
from torch.utils.tensorboard import SummaryWriter

root_detection = "./data/detect/test/"
video_dir = "../DATASET/test/"


def f1_test(video_name, idx_list, writer):
    video_path = os.path.join(video_dir, video_name + ".mp4")
    origin_detection_path = os.path.join(root_detection, video_name + "/labels")
    if not os.path.exists(origin_detection_path):
        command = ["--weights", "models/yolov5s6.pt", "--source", video_path, "--project", root_detection, "--name", video_name, "--save-txt", "--save-conf", "--nosave"]
        inference(command)

    origin_list = os.listdir(origin_detection_path)
    num_file = min(len(origin_list), len(idx_list))
    print(len(origin_list))
    print(len(idx_list))
    
    total_F1 = 0
    for i in range(num_file):
        skip_idx = idx_list[i]
        if skip_idx == -1: skip_idx = 0
        
        #print("idx: ", i, "skip_idx: ", skip_idx)
        origin, skip = origin_list[i], origin_list[skip_idx]
        origin_file = os.path.join(origin_detection_path, origin)
        skip_file = os.path.join(origin_detection_path, skip)
        F1_score = get_F1(origin_file, skip_file)
        
        if writer is not None:
            writer.add_scalar("F1_score/changes", F1_score, i)
        total_F1 += F1_score

    if writer is not None:
        writer.add_scalar("F1_score/total", total_F1, 1)
        writer.add_scalar("F1_score/average", total_F1/num_file, 1)

    F1_score = total_F1/num_file

    return F1_score


def make_idx_to_fraction_change_reducto(idx_list):
    segment = 120
    

def make_idx_to_fraction_change(idx_list, writer, interval=30):
    
    send_frame = 0
    for i in range(len(idx_list)):
        skip_idx = idx_list[i]
        if skip_idx == -1: skip_idx = 0
        
        #print("idx: ", i, "skip_idx: ", skip_idx)
        if i == skip_idx:
            send_frame += 1
    

    Fraction = send_frame/len(idx_list)
    
    return Fraction


def save_parameters_to_csv(data, prefix, f1_score, fraction):
    csv_file_path = 'test_exp_result.csv'
    
    existing_data = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                existing_data.append(row)

    new_row = [data, prefix, str(f1_score), str(fraction)]
    existing_data.append(new_row)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(existing_data)
        
        
if __name__ == "__main__":
    matching = {
        'JK': 'JK-1',
        'SD': 'SD-1',
        'JN': 'JN',
    }
    
    exp_matching = {}
    with open('matching.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                key = row[0]
                value = row[1]
                exp_matching[key] = value
    
    #path_list = ['config/test_config_LRLO.csv', 'config/test_config_V_effect.csv']
    path_list = ['config/test_config_LRLO_re_re.csv', 'config/test_config_LRLO_V_effect.csv', 'config/test_config_448_LRLO.csv']
    video_name = ['JK', 'SD']
    
    
    for config_path in path_list:
        print("In test config", config_path)
        
        with open(config_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            for row in reader:
                value = list(row)
                prefix = value[0]
                if value[2] != 'LRLO':
                    continue
                    idx_list_2d = eval(value[-2])
                    idx_list = [item for sublist in idx_list_2d for item in sublist]
                    video = value[3].split("/")[3]
                    if video == 'JN-1':
                        video = 'JN'
                    video_path = os.path.join(video_dir, video + ".mp4")
                    origin_detection_path = os.path.join(root_detection, video + "/labels")
                    
                    if not os.path.exists(origin_detection_path):
                        command = ["--weights", "models/yolov5s6.pt", "--source", video_path, "--project", root_detection, "--name", video, "--save-txt", "--save-conf", "--nosave"]
                        inference(command)
                    origin_list = os.listdir(origin_detection_path)
                    fraction = len(idx_list) / len(origin_list)
                    
                else:
                    idx_list = eval(value[-2])
                    print(idx_list[0], type(idx_list[0]))
                    idx_list = list(map(int, value[-2].strip('[]').split(',')))
                    video = value[3].split("/")[-1].split(".")[0]
                    f1_score = f1_test(video, idx_list, None)
                    fraction = make_idx_to_fraction_change(idx_list, None)
                
                fraction_jetson = value[-1]
                
                print("prefix:", prefix, "\t(fraction:", fraction, "*jetson score:", fraction_jetson, ")\n")
                save_parameters_to_csv(video, prefix, f1_score, fraction)