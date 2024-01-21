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


def save_parameters_to_csv(start_time:str, conf:Dict[str, Union[str, int, bool, float]], train:bool):
    if train: csv_file_path = 'train_config.csv'
    else: csv_file_path = 'test_config.csv'
    
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
