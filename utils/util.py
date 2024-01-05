import os
import csv
import argparse
from typing import Tuple, Union, Dict

def str2bool(v) :
    if isinstance(v, bool) :
        return v
    if v.lower() in ('true', 'yes', 't') :
        return True
    elif v.lower() in ('false', 'no', 'f') :
        return False
    else :
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_parameters_to_csv(start_time:str, output_path:str, conf:Dict[str, Union[str, int, bool, float]]):
    csv_file_path = 'config.csv'

    with open(csv_file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([])
        # 파라미터 딕셔너리를 CSV 파일에 저장
        for key, value in conf.items():
            csv_writer.writerow([key, value])


def save_parameters_to_csv(start_time:str, conf:Dict[str, Union[str, int, bool, float]]):
    csv_file_path = 'config.csv'
    
    existing_data = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                existing_data.append(row)

    new_row = [start_time] + [str(conf[key]) for key in conf.keys()]
    existing_data.append(new_row)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(existing_data)
